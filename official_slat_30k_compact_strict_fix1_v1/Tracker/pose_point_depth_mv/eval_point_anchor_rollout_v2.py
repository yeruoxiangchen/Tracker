#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch
from scipy.ndimage import label

TRACKER_ROOT = Path(__file__).resolve().parents[1]
for path in (
    TRACKER_ROOT,
    TRACKER_ROOT / "ReconViaGen",
    TRACKER_ROOT / "ReconViaGen" / "wheels" / "vggt",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from pose_point_depth_mv.eval_local_target_probe import (  # noqa: E402
    object_balanced,
    parse_csv_ints,
    summarize,
)
from pose_point_depth_mv.point_anchor_v2 import (  # noqa: E402
    ACTIVE_INDEX,
    OCCUPANCY_INDEX,
    POINT_ANCHOR_CHECKPOINT_VERSION,
    POINT_CONTROL_NAMES,
    PointAnchorCacheDataset,
    PointAnchorProbe,
    load_point_probe_state,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (  # noqa: E402
    install_unused_model_stubs,
)


ROLLOUT_VERSION = "pose_point_depth_mv.point_anchor_rollout.v2"
DECODE_NAMES = ("threshold_0", "topk_target_oracle_count")
BRANCH_NAMES = ("stock", "correct", *POINT_CONTROL_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-noise SS rollout audit for Point-anchor V2."
    )
    parser.add_argument("--point_cache_manifest", required=True)
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


def parse_interval(text: str) -> tuple[float, float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(values) != 2 or not 0.0 <= values[0] <= values[1] <= 1.0:
        raise ValueError(f"invalid CFG interval={text!r}")
    return values[0], values[1]


def rollout_noise_seed(uid: str, seed: int) -> int:
    digest = hashlib.sha256(f"{uid}:{int(seed)}:point-rollout-v2".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def timestep_pairs(steps: int, rescale_t: float) -> list[tuple[float, float]]:
    if int(steps) <= 0 or float(rescale_t) <= 0.0:
        raise ValueError("steps and rescale_t must be positive")
    sequence = np.linspace(1.0, 0.0, int(steps) + 1)
    sequence = float(rescale_t) * sequence / (
        1.0 + (float(rescale_t) - 1.0) * sequence
    )
    return [
        (float(sequence[index]), float(sequence[index + 1]))
        for index in range(int(steps))
    ]


def guided_stock_velocity(
    sampler: Any,
    flow: torch.nn.Module,
    x_t: torch.Tensor,
    t: float,
    condition: torch.Tensor,
    negative_condition: torch.Tensor,
    *,
    cfg_strength: float,
    cfg_interval: tuple[float, float],
    guidance_rescale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return positive, negative, and native guided stock velocity."""

    t_tensor = torch.full(
        (x_t.shape[0],), 1000.0 * float(t), device=x_t.device, dtype=torch.float32
    )
    positive = flow(x_t, t_tensor, condition)
    use_cfg = cfg_interval[0] <= float(t) <= cfg_interval[1]
    effective_strength = float(cfg_strength) if use_cfg else 1.0
    if effective_strength == 1.0:
        negative = positive
        return positive, negative, positive
    negative = flow(x_t, t_tensor, negative_condition)
    guided = effective_strength * positive + (1.0 - effective_strength) * negative
    if float(guidance_rescale) > 0.0:
        x0_positive = sampler._pred_to_xstart(x_t, float(t), positive)
        x0_guided = sampler._pred_to_xstart(x_t, float(t), guided)
        dimensions = list(range(1, x0_positive.ndim))
        std_positive = x0_positive.std(dim=dimensions, keepdim=True)
        std_guided = x0_guided.std(dim=dimensions, keepdim=True)
        x0_rescaled = x0_guided * (std_positive / std_guided)
        x0 = (
            float(guidance_rescale) * x0_rescaled
            + (1.0 - float(guidance_rescale)) * x0_guided
        )
        guided = sampler._xstart_to_pred(x_t, float(t), x0)
    return positive, negative, guided


@torch.no_grad()
def rollout_stock_loop(
    sampler: Any,
    flow: torch.nn.Module,
    noise: torch.Tensor,
    condition: torch.Tensor,
    negative_condition: torch.Tensor,
    *,
    steps: int,
    cfg_strength: float,
    cfg_interval: tuple[float, float],
    rescale_t: float,
    guidance_rescale: float,
) -> torch.Tensor:
    sample = noise.clone()
    for t, t_previous in timestep_pairs(steps, rescale_t):
        _, _, velocity = guided_stock_velocity(
            sampler,
            flow,
            sample,
            t,
            condition,
            negative_condition,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            guidance_rescale=guidance_rescale,
        )
        sample = sample - (t - t_previous) * velocity
    return sample


@torch.no_grad()
def rollout_point_branches(
    sampler: Any,
    flow: torch.nn.Module,
    probe: PointAnchorProbe,
    noise: torch.Tensor,
    condition: torch.Tensor,
    negative_condition: torch.Tensor,
    evidences: torch.Tensor,
    correct_mask: torch.Tensor,
    *,
    steps: int,
    cfg_strength: float,
    cfg_interval: tuple[float, float],
    rescale_t: float,
    guidance_rescale: float,
    physical_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Roll out correct and controls in one batch, with residual added after CFG."""

    branch_count = int(evidences.shape[0])
    if branch_count != 1 + len(POINT_CONTROL_NAMES):
        raise ValueError("rollout evidence batch does not contain correct plus controls")
    sample = noise.expand(branch_count, *noise.shape[1:]).clone()
    cond = condition.expand(branch_count, *condition.shape[1:]).contiguous()
    neg_cond = negative_condition.expand(
        branch_count, *negative_condition.shape[1:]
    ).contiguous()
    masks = correct_mask.expand(branch_count, *correct_mask.shape[1:])
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
        t_tensor = torch.full(
            (branch_count,),
            1000.0 * float(t),
            device=sample.device,
            dtype=torch.float32,
        )
        delta, stats = probe(
            sample,
            positive,
            t_tensor,
            evidences,
            scale=float(physical_scale),
            active_mask_override=masks,
        )
        sample = sample - (t - t_previous) * (guided + delta)
        delta_rms_values.append(float(stats["delta_rms"].float().item()))
        delta_abs_max = max(delta_abs_max, float(stats["delta_abs_max"].float().item()))
        neutral_abs_max = max(
            neutral_abs_max, float(stats["neutral_abs_max"].float().item())
        )
    return sample, {
        "delta_rms_step_mean": float(np.mean(delta_rms_values)),
        "delta_rms_step_max": float(np.max(delta_rms_values)),
        "delta_abs_max": delta_abs_max,
        "direct_neutral_delta_abs_max": neutral_abs_max,
    }


def coord_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(value) for value in row[-3:]) for row in coords}


def coords_from_logits(logits: torch.Tensor, target_count: int) -> dict[str, np.ndarray]:
    threshold = torch.nonzero(logits > 0.0, as_tuple=False)
    count = min(max(1, int(target_count)), int(logits.numel()))
    flat = torch.topk(logits.reshape(-1), k=count, largest=True).indices
    x = flat // (64 * 64)
    remainder = flat % (64 * 64)
    y = remainder // 64
    z = remainder % 64
    topk = torch.stack((x, y, z), dim=-1)
    return {
        "threshold_0": threshold.detach().cpu().numpy().astype(np.int32),
        "topk_target_oracle_count": topk.detach().cpu().numpy().astype(np.int32),
    }


def occupancy(coords: np.ndarray) -> np.ndarray:
    output = np.zeros((64, 64, 64), dtype=np.bool_)
    if coords.size:
        xyz = coords[:, -3:].astype(np.int64)
        valid = ((xyz >= 0) & (xyz < 64)).all(axis=1)
        xyz = xyz[valid]
        output[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
    return output


def overlap_metrics(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    pred = prediction & mask
    truth = target & mask
    intersection = int((pred & truth).sum())
    pred_count = int(pred.sum())
    target_count = int(truth.sum())
    union = pred_count + target_count - intersection
    precision = float(intersection / pred_count) if pred_count else 0.0
    recall = float(intersection / target_count) if target_count else 1.0
    return {
        "pred_count": pred_count,
        "target_count": target_count,
        "intersection": intersection,
        "iou": float(intersection / union) if union else 1.0,
        "precision": precision,
        "recall": recall,
        "f1": float(2.0 * precision * recall / (precision + recall))
        if precision + recall > 0.0
        else 0.0,
        "coord_count_ratio": float(pred_count / target_count) if target_count else 0.0,
    }


def component_metrics(prediction: np.ndarray) -> dict[str, float | int]:
    labels, count = label(prediction.astype(np.uint8))
    sizes = np.bincount(labels.reshape(-1))[1:] if count else np.zeros((0,))
    total = int(prediction.sum())
    return {
        "component_count": int(count),
        "largest_component_ratio": float(sizes.max() / total)
        if total and sizes.size
        else 0.0,
    }


def anchor_cell_hit_rate(prediction: np.ndarray, anchor_cells: np.ndarray) -> float:
    coarse = prediction.reshape(16, 4, 16, 4, 16, 4).any(axis=(1, 3, 5))
    denominator = int(anchor_cells.sum())
    return float((coarse & anchor_cells).sum() / denominator) if denominator else 0.0


def evaluate_coords(
    coords: np.ndarray,
    target: np.ndarray,
    active64: np.ndarray,
    anchor_cells: np.ndarray,
) -> dict[str, Any]:
    prediction = occupancy(coords)
    truth = occupancy(target)
    whole = np.ones_like(active64, dtype=np.bool_)
    result = {
        "global": overlap_metrics(prediction, truth, whole),
        "local": overlap_metrics(prediction, truth, active64),
        "outside": overlap_metrics(prediction, truth, ~active64),
        "correct_anchor_cell_hit_rate": anchor_cell_hit_rate(
            prediction, anchor_cells
        ),
    }
    result["global"].update(component_metrics(prediction))
    return result


def balanced_values(
    rows: list[dict[str, Any]],
    function: Callable[[dict[str, Any]], float],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    values = [
        {"object_uid": row["object_uid"], "value": float(function(row))}
        for row in rows
    ]
    return object_balanced(values, "value", bootstrap_samples=bootstrap_samples)


def branch_metric(
    row: dict[str, Any], branch: str, scope: str, metric: str
) -> float:
    return float(row["branches"][branch]["threshold_0"][scope][metric])


def compare_branches(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    pairs = {
        "global_iou": ("global", "iou"),
        "global_precision": ("global", "precision"),
        "global_recall": ("global", "recall"),
        "local_iou": ("local", "iou"),
        "local_precision": ("local", "precision"),
        "local_recall": ("local", "recall"),
        "outside_iou": ("outside", "iou"),
        "largest_component_ratio": ("global", "largest_component_ratio"),
        "component_count": ("global", "component_count"),
    }
    result = {
        name: balanced_values(
            rows,
            lambda row, scope=scope, metric=metric: branch_metric(
                row, left, scope, metric
            )
            - branch_metric(row, right, scope, metric),
            bootstrap_samples=bootstrap_samples,
        )
        for name, (scope, metric) in pairs.items()
    }
    result["correct_anchor_cell_hit_rate"] = balanced_values(
        rows,
        lambda row: float(
            row["branches"][left]["threshold_0"]["correct_anchor_cell_hit_rate"]
        )
        - float(
            row["branches"][right]["threshold_0"]["correct_anchor_cell_hit_rate"]
        ),
        bootstrap_samples=bootstrap_samples,
    )
    return result


def check_positive(metric: dict[str, Any], min_win: float) -> bool:
    return (
        float(metric["object"]["mean"]) > 0.0
        and float(metric["object"]["median"]) > 0.0
        and float(metric["object_win_rate"]) >= float(min_win)
        and float(metric["object_bootstrap_95_ci"][0]) > 0.0
    )


def load_models(
    pretrained: str, device: torch.device
) -> tuple[Any, torch.nn.Module, torch.nn.Module, dict[str, Any], dict[str, Any]]:
    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(pretrained)
    sampler = pipeline.sparse_structure_sampler
    defaults = dict(pipeline.sparse_structure_sampler_params)
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    for module in (flow, decoder):
        for parameter in module.parameters():
            parameter.requires_grad = False
    schema = {
        "resolution": int(flow.resolution),
        "in_channels": int(flow.in_channels),
        "out_channels": int(flow.out_channels),
        "flow_trainable_parameters": int(
            sum(parameter.numel() for parameter in flow.parameters() if parameter.requires_grad)
        ),
        "decoder_trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in decoder.parameters()
                if parameter.requires_grad
            )
        ),
    }
    del pipeline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sampler, flow, decoder, defaults, schema


def write_markdown(report: dict[str, Any], path: Path) -> None:
    stock = report["comparisons"]["correct_vs_stock"]
    lines = [
        "# Point-only Local-anchor V2 Fixed-noise Rollout",
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
            json.dumps(stock, indent=2),
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
        raise ValueError("point-anchor rollout evaluation requires CUDA")
    seeds = parse_csv_ints(args.seeds)
    cfg_interval = parse_interval(args.cfg_interval)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("format") != POINT_ANCHOR_CHECKPOINT_VERSION:
        raise ValueError("unexpected point-anchor checkpoint format")
    saved_args = checkpoint.get("args", {})
    if saved_args.get("pretrained") != args.pretrained:
        raise RuntimeError("point-anchor pretrained configuration mismatch")
    if float(saved_args.get("physical_scale", float("nan"))) != float(
        args.physical_scale
    ):
        raise RuntimeError("rollout physical_scale differs from training")

    dataset = PointAnchorCacheDataset(args.point_cache_manifest, indices=args.indices)
    model_summary = checkpoint.get("model_summary", {})
    if str(model_summary.get("cache_config_hash")) != dataset.config_hash:
        raise RuntimeError("point-anchor checkpoint/cache hash mismatch")
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
                f"rollout {key}={expected!r} differs from native default={defaults.get(key)!r}"
            )
    if float(args.guidance_rescale) != 0.0:
        raise ValueError("strict native rollout audit requires guidance_rescale=0")

    probe = PointAnchorProbe(rank=int(saved_args["rank"])).to(device).eval()
    load_point_probe_state(probe, checkpoint["model_trainable_state"])
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
        correct_evidence = sample["point_correct_evidence"].unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        controls = {
            name: sample["point_control_evidence"][name]
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
            for name in POINT_CONTROL_NAMES
        }
        evidences = torch.cat(
            (correct_evidence, *(controls[name] for name in POINT_CONTROL_NAMES)),
            dim=0,
        )
        correct_mask = probe.active_mask(correct_evidence).to(device=device)
        active16 = correct_mask[0, 0].detach().cpu().numpy().astype(np.bool_)
        active64 = np.repeat(np.repeat(np.repeat(active16, 4, 0), 4, 1), 4, 2)
        anchor_cells = (
            correct_evidence[0, OCCUPANCY_INDEX]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            > 0.5
        )
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
            if stock_audit is None:
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
                stock_audit = {
                    "latent_max_abs_diff": float(
                        (native_stock.float() - custom_stock.float()).abs().max().item()
                    ),
                    "latent_equal": bool(torch.equal(native_stock, custom_stock)),
                }

            modified, rollout_stats = rollout_point_branches(
                sampler,
                flow,
                probe,
                noise,
                condition,
                negative_condition,
                evidences,
                correct_mask,
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
            latents = torch.cat((native_stock, modified), dim=0)
            logits = decoder(
                latents.to(dtype=next(decoder.parameters()).dtype)
            ).float()[:, 0]
            branch_results: dict[str, Any] = {}
            for branch_index, branch in enumerate(BRANCH_NAMES):
                decoded = coords_from_logits(logits[branch_index], len(target_coords))
                branch_results[branch] = {
                    name: evaluate_coords(
                        coords, target_coords, active64, anchor_cells
                    )
                    for name, coords in decoded.items()
                }
            if stock_audit is not None and "threshold_coord_equal" not in stock_audit:
                # Decode both identical stock latents in the same batch. FP16 decoder
                # kernels can select a different threshold-boundary path when batch
                # size changes, which is not a sampler-equivalence failure.
                audit_logits = decoder(
                    torch.cat((native_stock, custom_stock), dim=0).to(
                        dtype=next(decoder.parameters()).dtype
                    )
                ).float()[:, 0]
                native_coords = coords_from_logits(
                    audit_logits[0], len(target_coords)
                )
                custom_coords = coords_from_logits(
                    audit_logits[1], len(target_coords)
                )
                stock_audit.update(
                    {
                        "threshold_coord_equal": coord_set(
                            native_coords["threshold_0"]
                        )
                        == coord_set(custom_coords["threshold_0"]),
                        "topk_coord_equal": coord_set(
                            native_coords["topk_target_oracle_count"]
                        )
                        == coord_set(custom_coords["topk_target_oracle_count"]),
                    }
                )
            rows.append(
                {
                    "source_index": int(sample["point_source_index"]),
                    "uid": uid,
                    "object_uid": object_uid,
                    "noise_seed": int(seed),
                    "active_ratio": float(active16.mean()),
                    "anchor_cell_count": int(anchor_cells.sum()),
                    "rollout_stats": rollout_stats,
                    "branches": branch_results,
                }
            )
            print(
                f"[point_anchor_rollout] {index + 1}/{count} seed={seed} uid={uid} "
                f"stock_local={branch_results['stock']['threshold_0']['local']['iou']:.6f} "
                f"correct_local={branch_results['correct']['threshold_0']['local']['iou']:.6f}",
                flush=True,
            )

    if stock_audit is None:
        raise RuntimeError("rollout produced no stock audit")
    correct_stock = compare_branches(
        rows, "correct", "stock", bootstrap_samples=int(args.bootstrap_samples)
    )
    correct_controls = {
        name: compare_branches(
            rows, "correct", name, bootstrap_samples=int(args.bootstrap_samples)
        )
        for name in POINT_CONTROL_NAMES
    }
    seed_local = {
        str(seed): summarize(
            [
                branch_metric(row, "correct", "local", "iou")
                - branch_metric(row, "stock", "local", "iou")
                for row in rows
                if int(row["noise_seed"]) == int(seed)
            ]
        )
        for seed in seeds
    }
    positive_seed_count = sum(float(item["mean"]) > 0.0 for item in seed_local.values())
    checks = {
        "native_stock_rollout_bit_exact": bool(stock_audit["latent_equal"])
        and stock_audit["latent_max_abs_diff"] == 0.0
        and bool(stock_audit.get("threshold_coord_equal"))
        and bool(stock_audit.get("topk_coord_equal")),
        "direct_non_anchor_delta_exact_zero": direct_neutral_max == 0.0,
        "object_disjoint_if_fresh": args.split_name != "fresh48" or not overlap,
        "correct_local_iou_beats_stock": check_positive(
            correct_stock["local_iou"], args.min_object_win_rate
        ),
        "correct_anchor_hit_beats_stock": check_positive(
            correct_stock["correct_anchor_cell_hit_rate"], args.min_object_win_rate
        ),
        "correct_local_iou_beats_every_control": all(
            check_positive(item["local_iou"], args.min_object_win_rate)
            for item in correct_controls.values()
        ),
        "positive_noise_seed_count": positive_seed_count
        >= int(args.min_positive_seed_count),
        "global_iou_preserved": float(correct_stock["global_iou"]["object"]["mean"])
        >= -float(args.max_global_iou_mean_degradation),
        "global_precision_preserved": float(
            correct_stock["global_precision"]["object"]["mean"]
        )
        >= -float(args.max_global_precision_mean_degradation),
        "outside_iou_preserved": float(correct_stock["outside_iou"]["object"]["mean"])
        >= -float(args.max_outside_iou_mean_degradation),
        "component_count_bounded": float(
            correct_stock["component_count"]["object"]["mean"]
        )
        <= float(args.max_component_count_mean_increase),
        "largest_component_ratio_preserved": float(
            correct_stock["largest_component_ratio"]["object"]["mean"]
        )
        >= -float(args.max_largest_component_ratio_mean_degradation),
    }
    report = {
        "format": ROLLOUT_VERSION,
        "stage": "Point-only local-anchor v2 fixed-noise SS rollout",
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
            "probe_stock_input": "positive conditional stock velocity",
            "sampler_velocity": "native guided stock velocity plus point delta",
            "fixed_correct_mask": True,
            "branch_batching": "correct plus six controls",
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
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": checks,
                "correct_vs_stock_local_iou": correct_stock["local_iou"],
                "positive_noise_seed_count": positive_seed_count,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.fail_on_decision and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
