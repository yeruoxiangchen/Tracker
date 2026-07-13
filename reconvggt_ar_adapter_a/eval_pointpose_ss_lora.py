#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from scipy.ndimage import label

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reconvggt_ar_adapter_a.pointpose_ss_condition import (  # noqa: E402
    load_partial_state,
    lora_disabled,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    PointPoseCacheDataset,
    build_models,
    encode_frozen_features,
    rgba_images,
    validate_checkpoint_config,
)


def coord_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    if not coords.size:
        return set()
    return {tuple(int(value) for value in row[-3:]) for row in coords}


def overlap(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred_set = coord_set(pred)
    target_set = coord_set(target)
    intersection = len(pred_set & target_set)
    union = len(pred_set | target_set)
    return {
        "pred_count": len(pred_set),
        "target_count": len(target_set),
        "intersection": intersection,
        "iou": float(intersection / union) if union else 1.0,
        "recall": float(intersection / len(target_set)) if target_set else 1.0,
        "precision": float(intersection / len(pred_set)) if pred_set else 0.0,
        "coord_count_ratio": float(len(pred_set) / len(target_set)) if target_set else 0.0,
    }


def component_stats(coords: np.ndarray) -> dict[str, float | int]:
    occupancy = np.zeros((64, 64, 64), dtype=np.uint8)
    if coords.size:
        xyz = coords[:, -3:].astype(np.int64)
        valid = ((xyz >= 0) & (xyz < 64)).all(axis=1)
        xyz = xyz[valid]
        occupancy[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = 1
    labels, count = label(occupancy)
    sizes = np.bincount(labels.reshape(-1))[1:] if count else np.zeros((0,), dtype=np.int64)
    total = int(occupancy.sum())
    return {
        "component_count": int(count),
        "largest_component_ratio": float(sizes.max() / total) if total and sizes.size else 0.0,
    }


def decode_coords(decoder, latent: torch.Tensor, target_count: int) -> dict[str, np.ndarray]:
    with torch.no_grad():
        logits = decoder(latent.to(dtype=next(decoder.parameters()).dtype)).float()[0, 0]
    threshold = torch.nonzero(logits > 0, as_tuple=False).detach().cpu().numpy().astype(np.int32)
    k = min(max(1, int(target_count)), int(logits.numel()))
    flat = torch.topk(logits.reshape(-1), k=k, largest=True).indices
    x = flat // (64 * 64)
    remainder = flat % (64 * 64)
    y = remainder // 64
    z = remainder % 64
    topk = torch.stack((x, y, z), dim=-1).detach().cpu().numpy().astype(np.int32)
    return {"threshold_0": threshold, "topk_target_oracle_count": topk}


METRIC_NAMES = (
    "iou",
    "recall",
    "precision",
    "pred_count",
    "coord_count_ratio",
    "component_count",
    "largest_component_ratio",
)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"sample_count": len(rows)}
    if not rows:
        return output
    sources = sorted(rows[0]["sources"])
    for source in sources:
        for decode in ("threshold_0", "topk_target_oracle_count"):
            metrics = [row["sources"][source][decode] for row in rows]
            key = f"{source}/{decode}"
            output[key] = {
                name: float(np.mean([float(metric[name]) for metric in metrics]))
                for name in METRIC_NAMES
            }
    return output


def object_balanced_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    objects: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        objects.setdefault(str(row["object_uid"]), []).append(row)
    per_object = [summarize(group) for group in objects.values()]
    output: dict[str, Any] = {"object_count": len(per_object)}
    if not per_object:
        return output
    keys = sorted(set.intersection(*(set(item) for item in per_object)) - {"sample_count"})
    for key in keys:
        output[key] = {
            metric: float(np.mean([item[key][metric] for item in per_object]))
            for metric in METRIC_NAMES
        }
    return output


def comparison_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = {
        "correct_minus_stock": ("pointpose_correct", "stock_native"),
        "correct_minus_image_only": ("pointpose_correct", "lora_image_only"),
        "correct_minus_shuffled": ("pointpose_correct", "pointpose_shuffled"),
        "correct_minus_zero": ("pointpose_correct", "pointpose_zero"),
    }
    output: dict[str, Any] = {}
    for label_name, (left, right) in comparisons.items():
        deltas = np.asarray(
            [
                row["sources"][left]["threshold_0"]["iou"]
                - row["sources"][right]["threshold_0"]["iou"]
                for row in rows
            ],
            dtype=np.float64,
        )
        output[label_name] = {
            "mean_iou_delta": float(deltas.mean()),
            "median_iou_delta": float(np.median(deltas)),
            "p25_iou_delta": float(np.quantile(deltas, 0.25)),
            "p75_iou_delta": float(np.quantile(deltas, 0.75)),
            "positive_win_rate": float((deltas > 0).mean()),
            "negative_degradation_rate": float((deltas < 0).mean()),
        }
    return output


def parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds must contain unique integers, got {text!r}")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-noise evaluation for PointPose SS Flow LoRA.")
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg_strength", type=float, default=7.5)
    parser.add_argument("--guidance_rescale", type=float, default=0.5)
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--physical_scale", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--physical_hidden_dim", type=int, default=256)
    parser.add_argument("--physical_heads", type=int, default=8)
    parser.add_argument("--bridge_train_last_blocks", type=int, default=0)
    args = parser.parse_args()
    args.gradient_checkpointing = False
    args.occupancy_weight = 0.0

    seeds = parse_seeds(args.seeds)
    random.seed(seeds[0])
    np.random.seed(seeds[0])
    torch.manual_seed(seeds[0])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_amp_dtype = checkpoint.get("args", {}).get("amp_dtype")
    if saved_amp_dtype not in {"fp16", "bf16", "none"}:
        raise RuntimeError(
            f"checkpoint has invalid or missing amp_dtype: {saved_amp_dtype!r}"
        )
    args.amp_dtype = str(saved_amp_dtype)
    validate_checkpoint_config(checkpoint, args, weights_only=False)

    device = torch.device(args.device)
    pipeline, model, _unused_decoder, model_summary = build_models(args, device)
    load_info = load_partial_state(
        model, checkpoint["model_trainable_state"], require_all_trainable=True
    )
    model.eval()
    decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad = False

    dataset = PointPoseCacheDataset(args.cache_manifest, indices=args.indices)
    count = len(dataset) if int(args.max_samples) <= 0 else min(len(dataset), int(args.max_samples))
    rows: list[dict[str, Any]] = []
    sampler_params = dict(pipeline.sparse_structure_sampler_params)
    sampler_params.update(
        {
            "steps": int(args.steps),
            "cfg_strength": float(args.cfg_strength),
            "guidance_rescale": float(args.guidance_rescale),
            "rescale_t": float(args.rescale_t),
        }
    )

    stock_rollout_audit = None
    for index in range(count):
        batch = dataset[index]
        images = rgba_images(batch["image_paths"], batch["mask_paths"], pipeline)
        aggregated, image_cond = encode_frozen_features(pipeline, images)
        physical = batch["physical_grid"].unsqueeze(0).to(device=device, dtype=torch.float32)
        shuffled_index = next(
            (
                candidate
                for offset in range(1, len(dataset) + 1)
                if (candidate := (index + offset) % len(dataset)) != index
                and dataset.samples[candidate].get("object_uid") != batch["object_uid"]
            ),
            None,
        )
        if shuffled_index is None:
            raise RuntimeError("pointpose_shuffled requires at least two distinct objects")
        shuffled_physical = dataset[shuffled_index]["physical_grid"].unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        zero_physical = torch.zeros_like(physical)
        target_coords = batch["target_coords"].numpy().astype(np.int32)
        with torch.no_grad():
            native_condition = pipeline.get_ss_cond(image_cond, aggregated, num_samples=1)
            cond_base = native_condition["cond"]
            # Reuse one native bridge result for every branch. Re-running the
            # low-precision bridge introduces small nondeterministic differences
            # and would also confound the point/pose ablation.
            cond_image, image_stats = model.physical_condition(cond_base, physical, scale=0.0)
            cond_correct, condition_stats = model.physical_condition(
                cond_base, physical, scale=float(args.physical_scale)
            )
            cond_shuffled, _ = model.physical_condition(
                cond_base, shuffled_physical, scale=float(args.physical_scale)
            )
            cond_zero, zero_stats = model.physical_condition(
                cond_base, zero_physical, scale=float(args.physical_scale)
            )
            if not torch.equal(native_condition["cond"], cond_image):
                raise RuntimeError("native stock condition differs from physical_scale=0 image condition")
            conditions = {
                "stock_native": native_condition["cond"],
                "lora_image_only": cond_image,
                "pointpose_correct": cond_correct,
                "pointpose_shuffled": cond_shuffled,
                "pointpose_zero": cond_zero,
            }
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(int(seed) + index * 1009)
            noise = torch.randn(
                (1, int(model.flow.in_channels), int(model.flow.resolution), int(model.flow.resolution), int(model.flow.resolution)),
                generator=generator, device=device, dtype=torch.float32,
            )
            latents: dict[str, torch.Tensor] = {}
            with torch.no_grad(), lora_disabled(model.flow):
                latents["stock_native"] = pipeline.sparse_structure_sampler.sample(
                    model.flow, noise.clone(), cond=conditions["stock_native"],
                    neg_cond=native_condition["neg_cond"], **sampler_params, verbose=False,
                ).samples
                if stock_rollout_audit is None:
                    repeated = pipeline.sparse_structure_sampler.sample(
                        model.flow, noise.clone(), cond=native_condition["cond"],
                        neg_cond=native_condition["neg_cond"], **sampler_params, verbose=False,
                    ).samples
                    stock_rollout_audit = {
                        "latent_max_abs_diff": float((latents["stock_native"] - repeated).abs().max().item()),
                    }
                    native_decoded = decode_coords(decoder, latents["stock_native"], len(target_coords))
                    repeated_decoded = decode_coords(decoder, repeated, len(target_coords))
                    pipeline_native_coords = pipeline.sample_sparse_structure(
                        native_condition,
                        num_samples=1,
                        sampler_params=sampler_params,
                        noise=noise.clone(),
                    ).detach().cpu().numpy().astype(np.int32)
                    stock_rollout_audit["threshold_coord_set_equal"] = bool(
                        coord_set(native_decoded["threshold_0"])
                        == coord_set(repeated_decoded["threshold_0"])
                    )
                    stock_rollout_audit["pipeline_native_threshold_coord_set_equal"] = bool(
                        coord_set(native_decoded["threshold_0"]) == coord_set(pipeline_native_coords)
                    )
                    stock_rollout_audit["oracle_count_coord_set_equal"] = bool(
                        coord_set(native_decoded["topk_target_oracle_count"])
                        == coord_set(repeated_decoded["topk_target_oracle_count"])
                    )
                    if (
                        stock_rollout_audit["latent_max_abs_diff"] != 0.0
                        or not stock_rollout_audit["threshold_coord_set_equal"]
                        or not stock_rollout_audit["pipeline_native_threshold_coord_set_equal"]
                        or not stock_rollout_audit["oracle_count_coord_set_equal"]
                    ):
                        raise RuntimeError(f"stock rollout equivalence failed: {stock_rollout_audit}")
            with torch.no_grad():
                for source in ("lora_image_only", "pointpose_correct", "pointpose_shuffled", "pointpose_zero"):
                    latents[source] = pipeline.sparse_structure_sampler.sample(
                        model.flow, noise.clone(), cond=conditions[source],
                        neg_cond=native_condition["neg_cond"], **sampler_params, verbose=False,
                    ).samples
            source_rows: dict[str, Any] = {}
            for source, latent in latents.items():
                decoded = decode_coords(decoder, latent, len(target_coords))
                source_rows[source] = {
                    decode_name: {**overlap(coords, target_coords), **component_stats(coords)}
                    for decode_name, coords in decoded.items()
                }
            rows.append(
                {
                    "index": index,
                    "seed": int(seed),
                    "uid": batch["uid"],
                    "object_uid": batch["object_uid"],
                    "shuffled_uid": dataset.samples[shuffled_index]["uid"],
                    "condition_delta_rms": float(condition_stats["delta_rms"].cpu().item()),
                    "zero_condition_delta_rms": float(zero_stats["delta_rms"].cpu().item()),
                    "image_only_delta_rms": float(image_stats["delta_rms"].cpu().item()),
                    "sources": source_rows,
                }
            )
            print(
                f"[pointpose_eval] {index + 1}/{count} seed={seed} uid={batch['uid']} "
                f"stock={source_rows['stock_native']['threshold_0']['iou']:.4f} "
                f"correct={source_rows['pointpose_correct']['threshold_0']['iou']:.4f}",
                flush=True,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "args": vars(args),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "load_info": load_info,
        "model": model_summary,
        "seeds": seeds,
        "stock_rollout_equivalence": stock_rollout_audit,
        "summary_sequence_weighted": summarize(rows),
        "summary_object_balanced": object_balanced_summary(rows),
        "rows": rows,
        "delta_summary": comparison_summary(rows),
        "comparison": {
            "stock_native": "Original ReconViaGen SS flow with LoRA disabled and native stock condition/neg_cond.",
            "lora_image_only": "Trained SS LoRA with physical residual disabled.",
            "pointpose_correct": "Trained SS LoRA plus aligned PointPose field.",
            "pointpose_shuffled": "Same image condition with another object's PointPose field.",
            "pointpose_zero": "Physical branch enabled with an all-zero PointPose field.",
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"summary": report["summary_sequence_weighted"], "delta": report["delta_summary"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
