#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ar_pose_trellis.pipeline import apply_lora_to_ss_flow  # noqa: E402
from trellis import models as trellis_models  # noqa: E402
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline  # noqa: E402

from trellis_point_prior_mv.common import (  # noqa: E402
    SparsePointPriorCond,
    coords_to_batched_occ,
    load_target_latent,
    parse_indices,
    partial_latent_stats,
    resolve_path,
    sparse_overlap_metrics,
    write_json,
)
from trellis_point_prior_mv.eval_sparse_inpaint import (  # noqa: E402
    make_prior_mode_coords,
    parse_topk_specs,
    topk_coords_from_logits,
)


def load_manifest(path: str | Path) -> tuple[dict, list[dict]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload, payload["samples"]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def point_overlap(prefix: str, pred_coords: np.ndarray, ref_coords: np.ndarray) -> Dict[str, float | int]:
    metrics = sparse_overlap_metrics(pred_coords, ref_coords)
    return {
        f"{prefix}_unique": metrics["target_unique"],
        f"{prefix}_intersection": metrics["intersection"],
        f"{prefix}_recall": metrics["target_recall"],
        f"{prefix}_precision": metrics["pred_precision"],
    }


def _coord_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    if coords.size == 0:
        return set()
    xyz = coords[:, -3:].astype(np.int32, copy=False)
    return set(map(tuple, xyz.tolist()))


def _latent_cell(coord: tuple[int, int, int], *, source_resolution: int, latent_resolution: int) -> tuple[int, int, int]:
    scale = max(1, int(source_resolution) // int(latent_resolution))
    return tuple(max(0, min(int(latent_resolution) - 1, int(v) // scale)) for v in coord)


def _set_region_metrics(prefix: str, pred_set: set[tuple[int, int, int]], target_set: set[tuple[int, int, int]]) -> Dict[str, float | int]:
    inter = pred_set & target_set
    union = pred_set | target_set
    return {
        f"{prefix}_pred_unique": len(pred_set),
        f"{prefix}_target_unique": len(target_set),
        f"{prefix}_intersection": len(inter),
        f"{prefix}_iou": float(len(inter) / len(union)) if union else 0.0,
        f"{prefix}_target_recall": float(len(inter) / len(target_set)) if target_set else 0.0,
        f"{prefix}_pred_precision": float(len(inter) / len(pred_set)) if pred_set else 0.0,
    }


def known_unknown_metrics(
    pred_coords: np.ndarray,
    target_coords: np.ndarray,
    prior_coords: np.ndarray,
    *,
    source_resolution: int = 64,
    latent_resolution: int = 16,
) -> Dict[str, float | int]:
    pred_set = _coord_set(pred_coords)
    target_set = _coord_set(target_coords)
    prior_set = _coord_set(prior_coords)
    known_cells = {
        _latent_cell(coord, source_resolution=source_resolution, latent_resolution=latent_resolution)
        for coord in prior_set
    }

    if known_cells:
        target_known = {
            coord
            for coord in target_set
            if _latent_cell(coord, source_resolution=source_resolution, latent_resolution=latent_resolution) in known_cells
        }
        pred_known = {
            coord
            for coord in pred_set
            if _latent_cell(coord, source_resolution=source_resolution, latent_resolution=latent_resolution) in known_cells
        }
    else:
        target_known = set()
        pred_known = set()

    target_unknown = target_set - target_known
    pred_unknown = pred_set - pred_known
    prior_target = prior_set & target_set
    prior_non_target = prior_set - target_set
    prior_non_target_pred = prior_non_target & pred_set
    false_positive_set = pred_set - target_set
    exact_unknown_target = target_set - prior_set
    exact_unknown_pred = pred_set - prior_set

    out: Dict[str, float | int] = {
        "known_cell_unique": len(known_cells),
        "prior_target_overlap_unique": len(prior_target),
        "prior_target_overlap_ratio": float(len(prior_target) / len(prior_set)) if prior_set else 0.0,
        "prior_non_target_unique": len(prior_non_target),
        "wrong_prior_leak_unique": len(prior_non_target_pred),
        "wrong_prior_leak_rate": float(len(prior_non_target_pred) / len(prior_non_target)) if prior_non_target else 0.0,
        "wrong_prior_leak_pred_fraction": float(len(prior_non_target_pred) / len(false_positive_set)) if false_positive_set else 0.0,
        "known_prior_recall": float(len(prior_set & pred_set) / len(prior_set)) if prior_set else 0.0,
        "known_prior_precision": float(len(prior_set & pred_set) / len(pred_set)) if pred_set else 0.0,
    }
    out.update(_set_region_metrics("known", pred_known, target_known))
    out.update(_set_region_metrics("unknown", pred_unknown, target_unknown))
    out.update(_set_region_metrics("exact_unknown", exact_unknown_pred, exact_unknown_target))
    return out


def prefix_metrics(prefix: str, metrics: Dict[str, float | int]) -> Dict[str, float | int]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def build_inpaint_condition(
    ss_encoder,
    ss_cond: SparsePointPriorCond,
    coords_np: np.ndarray,
    conf_np: np.ndarray,
    device: torch.device,
    *,
    known_conf_power: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = torch.from_numpy(coords_np.astype(np.int64))
    weights = torch.from_numpy(conf_np.astype(np.float32)).reshape(-1)
    if coords.numel():
        batch_col = torch.zeros((coords.shape[0], 1), dtype=torch.long)
        coords = torch.cat([batch_col, coords[:, -3:]], dim=1)
    else:
        coords = torch.zeros((0, 4), dtype=torch.long)
        weights = torch.zeros((0,), dtype=torch.float32)
    coords = coords.to(device)
    weights = weights.to(device)
    with torch.no_grad():
        occ = coords_to_batched_occ(coords, 1, resolution=64, device=device, dtype=next(ss_encoder.parameters()).dtype)
        raw_partial_latent = ss_encoder(occ, sample_posterior=False).to(torch.float32)
    mask, conf = partial_latent_stats(coords, 1, weights=weights, latent_resolution=raw_partial_latent.shape[-1], source_resolution=64, device=device)
    conf = conf.pow(float(known_conf_power)).clamp(0.0, 1.0)
    cond_partial_latent = raw_partial_latent * mask * conf
    cond = ss_cond(cond_partial_latent, mask, conf)
    return cond, raw_partial_latent, mask, conf


def inject_known_logits(logits: torch.Tensor, coords_np: np.ndarray, boost: float) -> torch.Tensor:
    if boost <= 0 or coords_np.size == 0:
        return logits
    out = logits.clone()
    coords = torch.from_numpy(coords_np[:, -3:].astype(np.int64)).to(out.device)
    valid = ((coords >= 0) & (coords < out.shape[-1])).all(dim=1)
    coords = coords[valid]
    if coords.numel() == 0:
        return out
    base = out.max().detach() + float(boost)
    d, h, w = coords[:, 0], coords[:, 1], coords[:, 2]
    out[0, 0, d, h, w] = torch.maximum(out[0, 0, d, h, w], base.expand_as(d).to(out.dtype))
    return out


@torch.no_grad()
def sample_latent_with_known_reinjection(
    pipeline,
    flow,
    cond: torch.Tensor,
    noise: torch.Tensor,
    partial_latent: torch.Tensor,
    latent_mask: torch.Tensor,
    confidence: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    sampler = pipeline.sparse_structure_sampler
    sampler_params = {**pipeline.sparse_structure_sampler_params}
    sampler_params.pop("steps", None)
    sampler_params.pop("verbose", None)
    rescale_t = float(sampler_params.pop("rescale_t", 1.0))
    sampler_cfg_strength = float(sampler_params.pop("cfg_strength", 1.0))
    cfg_strength = float(args.guidance_strength if args.guidance_strength is not None else sampler_cfg_strength)
    guidance_rescale = sampler_params.pop("guidance_rescale", None)
    steps = int(args.steps)
    model_kwargs = dict(sampler_params)
    model_kwargs.update(
        {
            "neg_cond": torch.zeros_like(cond),
            "cfg_strength": cfg_strength,
        }
    )
    if guidance_rescale is not None:
        model_kwargs["guidance_rescale"] = guidance_rescale
    sample = noise.clone()
    known_weight = (latent_mask * confidence if args.known_use_confidence else latent_mask).clamp(0.0, 1.0)
    clamp = (known_weight * float(args.known_latent_clamp_strength)).clamp(0.0, 1.0)
    clamp_start_t = float(getattr(args, "known_clamp_start_t", 1.0))
    known_noise = noise.clone()
    if args.known_latent_clamp_strength > 0 and args.clamp_initial_noise and 1.0 <= clamp_start_t + 1e-8:
        known_xt = sampler._xstart_to_x_t(partial_latent, 1.0, known_noise)
        sample = sample * (1.0 - clamp) + known_xt * clamp
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    for t, t_prev in list((t_seq[i], t_seq[i + 1]) for i in range(steps)):
        out = sampler.sample_once(
            flow,
            sample,
            float(t),
            float(t_prev),
            cond,
            **model_kwargs,
        )
        sample = out.pred_x_prev
        if args.known_latent_clamp_strength > 0 and float(t_prev) <= clamp_start_t + 1e-8:
            known_xt_prev = sampler._xstart_to_x_t(partial_latent, float(t_prev), known_noise)
            sample = sample * (1.0 - clamp) + known_xt_prev * clamp
    return sample


def build_models(args: argparse.Namespace, device: torch.device):
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.to(device)
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    for p in flow.parameters():
        p.requires_grad = False
    flow = apply_lora_to_ss_flow(flow, r=args.lora_rank, alpha=args.lora_alpha).to(device).eval()
    decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    encoder = trellis_models.from_pretrained(
        f"{args.weights}/ckpts/ss_enc_conv3d_16l8_fp16"
        if os.path.isdir(args.weights)
        else f"{args.weights}/ckpts/ss_enc_conv3d_16l8_fp16"
    ).to(device).eval()
    ss_cond = SparsePointPriorCond(
        latent_channels=args.latent_channels,
        cond_channels=args.cond_channels,
        grid_resolution=args.latent_grid_resolution,
    ).to(device).eval()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        state = state.get("state_dict", state)
        flow_state = {k.replace("ss_flow_model.", "", 1): v for k, v in state.items() if k.startswith("ss_flow_model.")}
        cond_state = {k.replace("ss_cond.", "", 1): v for k, v in state.items() if k.startswith("ss_cond.")}
        if flow_state:
            missing, unexpected = flow.load_state_dict(flow_state, strict=False)
            print(f"[eval_sparse_inpaint_stage2] flow load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        missing, unexpected = ss_cond.load_state_dict(cond_state, strict=False)
        print(f"[eval_sparse_inpaint_stage2] cond load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return pipeline, flow, decoder, encoder, ss_cond


def summarize(rows: list[dict]) -> dict:
    out: Dict[str, Any] = {}
    metric_keys = [
        "iou",
        "target_recall",
        "pred_precision",
        "prior_recall",
        "prior_precision",
        "pred_unique",
        "intersection",
        "known_iou",
        "known_target_recall",
        "known_pred_precision",
        "unknown_iou",
        "unknown_target_recall",
        "unknown_pred_precision",
        "exact_unknown_iou",
        "exact_unknown_target_recall",
        "known_prior_recall",
        "prior_target_overlap_ratio",
        "wrong_prior_leak_rate",
        "wrong_prior_leak_pred_fraction",
        "obs_known_iou",
        "obs_known_target_recall",
        "obs_known_pred_precision",
        "obs_unknown_iou",
        "obs_unknown_target_recall",
        "obs_unknown_pred_precision",
        "obs_exact_unknown_iou",
        "obs_exact_unknown_target_recall",
        "obs_known_prior_recall",
        "obs_prior_target_overlap_ratio",
        "obs_wrong_prior_leak_rate",
        "obs_wrong_prior_leak_pred_fraction",
    ]
    for mode in sorted({row["prior_mode"] for row in rows}):
        mode_rows = [r for r in rows if r["prior_mode"] == mode]
        out[mode] = {"count": len(mode_rows)}
        for key in metric_keys:
            vals = [float(r[key]) for r in mode_rows if key in r]
            out[mode][f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
            out[mode][f"{key}_median"] = float(np.median(vals)) if vals else 0.0

    def add_rank_summary(out_key: str, metric_key: str) -> None:
        rank_rows = []
        for sample_idx in sorted({row["sample_index"] for row in rows}):
            for topk_label in sorted({row["topk_label"] for row in rows if row["sample_index"] == sample_idx}):
                group = [r for r in rows if r["sample_index"] == sample_idx and r["topk_label"] == topk_label]
                correct = [r for r in group if r["prior_mode"] == "correct"]
                if not correct or any(metric_key not in r for r in group):
                    continue
                sorted_group = sorted(group, key=lambda r: float(r[metric_key]), reverse=True)
                rank = next(i + 1 for i, r in enumerate(sorted_group) if r["prior_mode"] == "correct")
                rank_rows.append({"sample_index": sample_idx, "topk_label": topk_label, "rank": rank, "top1": rank == 1})
        out[out_key] = {
            "count": len(rank_rows),
            "top1": int(sum(1 for r in rank_rows if r["top1"])),
            "top1_rate": float(np.mean([r["top1"] for r in rank_rows])) if rank_rows else 0.0,
            "rank_mean": float(np.mean([r["rank"] for r in rank_rows])) if rank_rows else 0.0,
            "rank_median": float(np.median([r["rank"] for r in rank_rows])) if rank_rows else 0.0,
        }

    add_rank_summary("correct_rank", "iou")
    add_rank_summary("correct_rank_unknown_iou", "unknown_iou")
    add_rank_summary("correct_rank_unknown_target_recall", "unknown_target_recall")
    add_rank_summary("correct_rank_obs_unknown_iou", "obs_unknown_iou")
    add_rank_summary("correct_rank_obs_unknown_target_recall", "obs_unknown_target_recall")

    paired: Dict[str, Any] = {}
    paired_metrics = [
        "iou",
        "target_recall",
        "unknown_iou",
        "unknown_target_recall",
        "wrong_prior_leak_rate",
        "obs_known_iou",
        "obs_known_target_recall",
        "obs_unknown_iou",
        "obs_unknown_target_recall",
        "obs_known_prior_recall",
        "obs_wrong_prior_leak_rate",
    ]
    topk_labels = sorted({row["topk_label"] for row in rows})
    for sample_idx in sorted({row["sample_index"] for row in rows}):
        for topk_label in topk_labels:
            group = [r for r in rows if r["sample_index"] == sample_idx and r["topk_label"] == topk_label]
            by_mode = {r["prior_mode"]: r for r in group}
            correct = by_mode.get("correct")
            if correct is None:
                continue
            for wrong_mode, wrong in by_mode.items():
                if wrong_mode == "correct":
                    continue
                key = f"{topk_label}/{wrong_mode}"
                bucket = paired.setdefault(key, {metric: [] for metric in paired_metrics})
                for metric in paired_metrics:
                    if metric in correct and metric in wrong:
                        bucket[metric].append(float(correct[metric]) - float(wrong[metric]))
    out["paired_delta"] = {}
    for key, values in sorted(paired.items()):
        out["paired_delta"][key] = {}
        for metric, deltas in values.items():
            out["paired_delta"][key][metric] = {
                "mean": float(np.mean(deltas)) if deltas else 0.0,
                "median": float(np.median(deltas)) if deltas else 0.0,
                "correct_wins": int(sum(1 for v in deltas if v > 0)),
                "count": len(deltas),
            }
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    payload, samples = load_manifest(args.manifest)
    indices = parse_indices(args.indices, len(samples))
    prior_root = payload.get("prior_root") or str(Path(args.manifest).parent)
    pipeline, flow, decoder, encoder, ss_cond = build_models(args, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.prior_modes.split(",") if m.strip()]
    topk_specs = parse_topk_specs(args.fixed_topk)
    rows = []
    for order, sample_idx in enumerate(indices):
        sample = samples[sample_idx]
        uid = str(sample.get("uid", sample_idx))
        _z, target_coords = load_target_latent(sample["ss_latent"])
        with np.load(resolve_path(prior_root, sample["prior_npz"])) as data:
            correct_prior = np.asarray(data["prior_coords"], dtype=np.int64)
            correct_conf = np.asarray(data["prior_conf"], dtype=np.float32) if "prior_conf" in data else np.ones((correct_prior.shape[0],), dtype=np.float32)
        rng = np.random.default_rng(args.seed + sample_idx * 1009)
        torch.manual_seed(args.seed + sample_idx * 9973)
        base_noise = torch.randn(1, flow.in_channels, int(flow.resolution), int(flow.resolution), int(flow.resolution), device=device)
        target_unique = len(set(map(tuple, target_coords[:, -3:].astype(np.int32).tolist())))
        for mode in modes:
            prior_coords, prior_conf, prior_source = make_prior_mode_coords(mode, correct_prior, correct_conf, len(target_coords), samples, prior_root, sample_idx, rng)
            cond, partial_latent, latent_mask, confidence = build_inpaint_condition(
                encoder,
                ss_cond,
                prior_coords,
                prior_conf,
                device,
                known_conf_power=args.known_conf_power,
            )
            latent = sample_latent_with_known_reinjection(
                pipeline,
                flow,
                cond,
                base_noise,
                partial_latent,
                latent_mask,
                confidence,
                args,
            )
            logits = decoder(latent)
            if logits.ndim != 5:
                raise ValueError(f"decoder returned bad logits shape {tuple(logits.shape)}")
            if logits.shape[1] != 1:
                logits = logits.max(dim=1, keepdim=True).values
            logits = inject_known_logits(logits.float(), prior_coords, args.known_logit_boost)
            for spec in topk_specs:
                topk = target_unique if spec == "target_unique" else int(spec)
                pred = topk_coords_from_logits(logits, topk)
                target_metrics = sparse_overlap_metrics(pred, target_coords)
                prior_metrics = point_overlap("prior", pred, prior_coords)
                region_metrics = known_unknown_metrics(
                    pred,
                    target_coords,
                    prior_coords,
                    source_resolution=64,
                    latent_resolution=args.latent_grid_resolution,
                )
                obs_region_metrics = prefix_metrics(
                    "obs_",
                    known_unknown_metrics(
                        pred,
                        target_coords,
                        correct_prior,
                        source_resolution=64,
                        latent_resolution=args.latent_grid_resolution,
                    ),
                )
                row = {
                    "sample_order": order,
                    "sample_index": sample_idx,
                    "uid": uid,
                    "prior_mode": mode,
                    "prior_source": prior_source,
                    "prior_points": int(prior_coords.shape[0]),
                    "topk_label": spec,
                    "topk": int(topk),
                    "latent_clamp_strength": float(args.known_latent_clamp_strength),
                    "known_logit_boost": float(args.known_logit_boost),
                    "known_use_confidence": int(bool(args.known_use_confidence)),
                    **target_metrics,
                    **prior_metrics,
                    **region_metrics,
                    **obs_region_metrics,
                }
                rows.append(row)
                print(
                    f"[eval_sparse_inpaint_stage2] idx={sample_idx} mode={mode} topk={spec} "
                    f"iou={row['iou']:.4f} target_recall={row['target_recall']:.4f} "
                    f"unknown_iou={row['unknown_iou']:.4f} "
                    f"unknown_recall={row['unknown_target_recall']:.4f} "
                    f"obs_unknown_iou={row['obs_unknown_iou']:.4f} "
                    f"obs_unknown_recall={row['obs_unknown_target_recall']:.4f} "
                    f"prior_recall={row['prior_recall']:.4f}",
                    flush=True,
                )
    report = {"args": vars(args), "summary": summarize(rows), "rows": rows}
    write_json(output_dir / "report.json", report)
    write_csv(output_dir / "report.csv", rows)
    print(f"[eval_sparse_inpaint_stage2] wrote {output_dir / 'report.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--indices", default="0-7")
    parser.add_argument("--prior_modes", default="correct,empty,shuffle,random,jitter")
    parser.add_argument("--fixed_topk", default="4096,8192,target_unique")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance_strength", type=float, default=1.0)
    parser.add_argument("--known_latent_clamp_strength", type=float, default=1.0)
    parser.add_argument("--known_clamp_start_t", type=float, default=1.0)
    parser.add_argument("--known_logit_boost", type=float, default=0.0)
    parser.add_argument("--known_conf_power", type=float, default=1.0)
    parser.add_argument("--known_use_confidence", action="store_true")
    parser.add_argument("--clamp_initial_noise", dest="clamp_initial_noise", action="store_true", default=True)
    parser.add_argument("--no_clamp_initial_noise", dest="clamp_initial_noise", action="store_false")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--latent_channels", type=int, default=8)
    parser.add_argument("--latent_grid_resolution", type=int, default=16)
    parser.add_argument("--cond_channels", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    main()
