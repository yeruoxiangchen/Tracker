#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
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

from reconvggt_ar_adapter_a.pointpose_patch_features import (  # noqa: E402
    PROJECTED_PATCH_FEATURE_NAMES,
    make_null_projected_patch_features,
)
from reconvggt_ar_adapter_a.pointpose_ss_condition import load_partial_state  # noqa: E402
from reconvggt_ar_adapter_a.sparse_anchor_flow import build_sparse_anchor_masks  # noqa: E402
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    PointPoseCacheDataset,
    encode_frozen_features,
    rgba_images,
)
from reconvggt_ar_adapter_a.train_stock_preserving_pointpose_bridge import (  # noqa: E402
    HardNegativeMiner,
    build_bridge_condition_inputs,
    build_models,
)


SOURCES = ("stock", "full_correct", "mask_only", "point_only", "full_shuffled", "null")


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "positive_rate": float((array > 0).mean()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def selected_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask.expand_as(values)]
    if selected.numel() == 0:
        raise RuntimeError("attribution mask is empty")
    return float(selected.float().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-checkpoint mask/point attribution for J1a.2-B1."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="0-15")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t", type=float, default=0.5)
    parser.add_argument("--physical_hidden_dim", type=int, default=128)
    parser.add_argument("--physical_heads", type=int, default=8)
    parser.add_argument("--bridge_last_blocks", type=int, default=1)
    parser.add_argument("--fusion_stages", default="0,1")
    parser.add_argument("--local_fusion_hidden_dim", type=int, default=128)
    parser.add_argument("--content_fusion_dim", type=int, default=128)
    parser.add_argument("--content_fusion_heads", type=int, default=4)
    parser.add_argument("--prior_confidence_min", type=float, default=0.25)
    parser.add_argument("--anchor_radius_16", type=int, default=1)
    parser.add_argument("--outside_visible_min", type=float, default=0.5)
    parser.add_argument("--outside_ratio_min", type=float, default=0.9)
    args = parser.parse_args()
    args.bridge_fusion_mode = "pose_guided_patch"
    args.gradient_checkpointing = False

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    pipeline, model, model_summary = build_models(args, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved = checkpoint.get("args", {})
    expected = {
        "pretrained": str(args.pretrained),
        "bridge_fusion_mode": "pose_guided_patch",
        "fusion_stages": str(args.fusion_stages),
        "physical_hidden_dim": int(args.physical_hidden_dim),
        "content_fusion_dim": int(args.content_fusion_dim),
    }
    mismatch = {
        key: {"checkpoint": saved.get(key), "current": value}
        for key, value in expected.items()
        if saved.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"B1 attribution checkpoint mismatch: {mismatch}")
    load_info = load_partial_state(
        model,
        checkpoint["model_trainable_state"],
        require_all_trainable=True,
    )
    model.eval()
    flow = model.flow.eval()
    decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad = False
    decoder_dtype = next(decoder.parameters()).dtype
    dataset = PointPoseCacheDataset(args.cache_manifest, indices=args.indices)
    negative_miner = HardNegativeMiner(dataset)
    count = len(dataset) if int(args.max_samples) <= 0 else min(len(dataset), int(args.max_samples))
    mask_index = PROJECTED_PATCH_FEATURE_NAMES.index("mask_patch_fraction")
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for index in range(count):
            sample = dataset[index]
            negative = dataset[negative_miner[index]]
            images = rgba_images(sample["image_paths"], sample["mask_paths"], pipeline)
            aggregated, image_cond = encode_frozen_features(pipeline, images)
            correct_input, shuffled_input, input_audit = build_bridge_condition_inputs(
                model, sample, negative, aggregated, device
            )
            null_input = make_null_projected_patch_features(correct_input)
            mask_only = null_input.clone()
            mask_only[:, :, mask_index] = correct_input[:, :, mask_index]
            point_only = correct_input.clone()
            point_only[:, :, mask_index] = 0.0
            inputs = {
                "full_correct": correct_input,
                "mask_only": mask_only,
                "point_only": point_only,
                "full_shuffled": shuffled_input,
                "null": null_input,
            }
            conditions: dict[str, torch.Tensor] = {}
            stock_condition = None
            for source, condition_input in inputs.items():
                paths = model.bridge_fusion.condition_paths(
                    aggregated,
                    image_cond,
                    condition_input,
                    physical_scale=1.0,
                )
                conditions[source] = paths.cond_fused
                if stock_condition is None:
                    stock_condition = paths.cond_stock
                elif not torch.equal(paths.cond_stock, stock_condition):
                    raise RuntimeError("stock bridge condition changed across attribution sources")
            assert stock_condition is not None
            conditions["stock"] = stock_condition

            physical = sample["physical_grid"].unsqueeze(0).to(device=device, dtype=torch.float32)
            target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
            masks = build_sparse_anchor_masks(
                physical,
                sample["target_coords"].to(device),
                prior_confidence_min=float(args.prior_confidence_min),
                anchor_radius_16=int(args.anchor_radius_16),
                outside_visible_min=float(args.outside_visible_min),
                outside_ratio_min=float(args.outside_ratio_min),
            )
            generator = torch.Generator(device=device).manual_seed(int(args.seed) * 1000003 + index)
            noise = torch.randn(target.shape, device=device, dtype=target.dtype, generator=generator)
            x_t, gt_velocity = pipeline.sparse_structure_sampler._get_model_gt(
                target, float(args.t), noise
            )
            t_tensor = torch.full((1,), 1000.0 * float(args.t), device=device)
            source_rows: dict[str, Any] = {}
            for source in SOURCES:
                velocity = flow(x_t, t_tensor, conditions[source])
                x0 = pipeline.sparse_structure_sampler._pred_to_xstart(
                    x_t, float(args.t), velocity
                )
                logits = decoder(x0.to(dtype=decoder_dtype)).float()
                probability = torch.sigmoid(logits)
                source_rows[source] = {
                    "flow_mse": float(F.mse_loss(velocity.float(), gt_velocity.float()).item()),
                    "positive_probability": selected_mean(probability, masks["positive64"]),
                    "outside_probability": selected_mean(probability, masks["negative64"]),
                    "condition_delta_rms": float(
                        (conditions[source].float() - stock_condition.float()).square().mean().sqrt().item()
                    ),
                }
            rows.append(
                {
                    "uid": str(sample["uid"]),
                    "object_uid": str(sample["object_uid"]),
                    "input_audit": input_audit,
                    "sources": source_rows,
                }
            )

    source_summary = {
        source: {
            metric: distribution([float(row["sources"][source][metric]) for row in rows])
            for metric in (
                "flow_mse",
                "positive_probability",
                "outside_probability",
                "condition_delta_rms",
            )
        }
        for source in SOURCES
    }
    gains = {}
    for source in ("full_correct", "mask_only", "point_only", "full_shuffled", "null"):
        gains[source] = {
            "stock_minus_flow": distribution([
                float(row["sources"]["stock"]["flow_mse"] - row["sources"][source]["flow_mse"])
                for row in rows
            ]),
            "positive_minus_stock": distribution([
                float(
                    row["sources"][source]["positive_probability"]
                    - row["sources"]["stock"]["positive_probability"]
                )
                for row in rows
            ]),
            "stock_minus_outside": distribution([
                float(
                    row["sources"]["stock"]["outside_probability"]
                    - row["sources"][source]["outside_probability"]
                )
                for row in rows
            ]),
        }
    full_gain = float(gains["full_correct"]["stock_minus_flow"]["mean"])
    mask_gain = float(gains["mask_only"]["stock_minus_flow"]["mean"])
    point_gain = float(gains["point_only"]["stock_minus_flow"]["mean"])
    report = {
        "format": "reconvggt.b1_sparse_attribution.v1",
        "args": vars(args),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "load_info": load_info,
        "model": model_summary,
        "source_summary": source_summary,
        "gains_vs_stock": gains,
        "attribution": {
            "full_correct_flow_gain": full_gain,
            "mask_only_flow_gain": mask_gain,
            "point_only_flow_gain": point_gain,
            "mask_fraction_of_full": mask_gain / max(abs(full_gain), 1.0e-12),
            "point_fraction_of_full": point_gain / max(abs(full_gain), 1.0e-12),
            "interpretation_only": True,
        },
        "rows": rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# B1 Sparse Input Attribution",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- samples: `{len(rows)}`",
        "",
        "| source | flow gain vs stock | positive gain | outside gain | condition delta RMS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source in ("full_correct", "mask_only", "point_only", "full_shuffled", "null"):
        lines.append(
            f"| {source} | {gains[source]['stock_minus_flow']['mean']:.8f} | "
            f"{gains[source]['positive_minus_stock']['mean']:.8f} | "
            f"{gains[source]['stock_minus_outside']['mean']:.8f} | "
            f"{source_summary[source]['condition_delta_rms']['mean']:.8f} |"
        )
    lines.extend(
        [
            "",
            f"- mask/full flow-gain ratio: `{report['attribution']['mask_fraction_of_full']:.6f}`",
            f"- point/full flow-gain ratio: `{report['attribution']['point_fraction_of_full']:.6f}`",
            "- This is an attribution diagnostic, not a checkpoint-selection gate.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["attribution"], indent=2), flush=True)


if __name__ == "__main__":
    main()
