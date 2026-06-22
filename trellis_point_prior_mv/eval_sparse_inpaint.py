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


def parse_topk_specs(spec: str) -> List[str]:
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower() in {"target", "target_unique", "gt"}:
            out.append("target_unique")
        else:
            value = int(part)
            if value <= 0:
                raise ValueError(f"topk must be > 0: {part}")
            out.append(str(value))
    return out


def topk_coords_from_logits(logits: torch.Tensor, topk: int) -> np.ndarray:
    if logits.ndim != 5:
        raise ValueError(f"expected logits [B,C,D,H,W], got {tuple(logits.shape)}")
    b, _c, d, h, w = logits.shape
    if b != 1:
        raise ValueError("eval currently expects batch size 1")
    flat = logits.reshape(1, -1)
    k = max(0, min(int(topk), flat.shape[1]))
    if k == 0:
        return np.zeros((0, 4), dtype=np.int32)
    _values, idx = torch.topk(flat, k=k, dim=1)
    idx = idx[0] % (d * h * w)
    z = idx // (h * w)
    y = (idx % (h * w)) // w
    x = idx % w
    coords = torch.stack([torch.zeros_like(z), z, y, x], dim=1).detach().cpu().numpy().astype(np.int32)
    return coords


def make_prior_mode_coords(
    mode: str,
    correct_coords: np.ndarray,
    correct_conf: np.ndarray,
    target_count: int,
    samples: list[dict],
    prior_root: str,
    index: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    mode = mode.lower()
    if mode == "correct":
        return correct_coords.copy(), correct_conf.copy(), "self"
    if mode == "empty":
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.float32), "empty"
    if mode == "random":
        count = max(1, int(correct_coords.shape[0]))
        coords = rng.integers(0, 64, size=(count, 3), dtype=np.int64)
        return coords, np.ones((coords.shape[0],), dtype=np.float32), "random"
    if mode == "jitter":
        if correct_coords.shape[0] == 0:
            return correct_coords.copy(), correct_conf.copy(), "self_jitter_empty"
        noise = rng.integers(-4, 5, size=correct_coords.shape, dtype=np.int64)
        return np.clip(correct_coords.astype(np.int64) + noise, 0, 63), correct_conf.copy(), "self_jitter"
    if mode in {"shuffle", "cross_sample"}:
        choices = [i for i in range(len(samples)) if i != index]
        other = int(rng.choice(choices)) if choices else index
        with np.load(resolve_path(prior_root, samples[other]["prior_npz"])) as data:
            coords = np.asarray(data["prior_coords"], dtype=np.int64)
            conf = np.asarray(data["prior_conf"], dtype=np.float32) if "prior_conf" in data else np.ones((coords.shape[0],), dtype=np.float32)
            return coords, conf, str(samples[other].get("uid", other))
    raise ValueError(f"unknown prior mode: {mode}")


def build_condition(
    ss_encoder,
    ss_cond: SparsePointPriorCond,
    coords_np: np.ndarray,
    conf_np: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
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
        partial_latent = ss_encoder(occ, sample_posterior=False).to(torch.float32)
    mask, conf = partial_latent_stats(coords, 1, weights=weights, latent_resolution=partial_latent.shape[-1], source_resolution=64, device=device)
    partial_latent = partial_latent * mask * conf
    return ss_cond(partial_latent, mask, conf)


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
            print(f"[eval_sparse_inpaint] flow load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        missing, unexpected = ss_cond.load_state_dict(cond_state, strict=False)
        print(f"[eval_sparse_inpaint] cond load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return pipeline, flow, decoder, encoder, ss_cond


@torch.no_grad()
def sample_logits(pipeline, flow, decoder, cond: torch.Tensor, noise: torch.Tensor, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    sampler_params = {**pipeline.sparse_structure_sampler_params, "steps": args.steps, "cfg_strength": args.guidance_strength}
    out = pipeline.sparse_structure_sampler.sample(
        flow,
        noise.clone(),
        cond=cond,
        neg_cond=torch.zeros_like(cond),
        **sampler_params,
        verbose=False,
    ).samples
    logits = decoder(out)
    if logits.ndim != 5:
        raise ValueError(f"decoder returned bad logits shape {tuple(logits.shape)}")
    if logits.shape[1] != 1:
        logits = logits.max(dim=1, keepdim=True).values
    return logits.float()


def summarize(rows: list[dict]) -> dict:
    out: Dict[str, Any] = {}
    for mode in sorted({row["prior_mode"] for row in rows}):
        mode_rows = [r for r in rows if r["prior_mode"] == mode]
        out[mode] = {"count": len(mode_rows)}
        for key in ["iou", "target_recall", "pred_precision", "pred_unique", "intersection"]:
            vals = [float(r[key]) for r in mode_rows if key in r]
            out[mode][f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
            out[mode][f"{key}_median"] = float(np.median(vals)) if vals else 0.0

    rank_rows = []
    for sample_idx in sorted({row["sample_index"] for row in rows}):
        for topk_label in sorted({row["topk_label"] for row in rows if row["sample_index"] == sample_idx}):
            group = [r for r in rows if r["sample_index"] == sample_idx and r["topk_label"] == topk_label]
            correct = [r for r in group if r["prior_mode"] == "correct"]
            if not correct:
                continue
            sorted_group = sorted(group, key=lambda r: r["iou"], reverse=True)
            rank = next(i + 1 for i, r in enumerate(sorted_group) if r["prior_mode"] == "correct")
            rank_rows.append({"sample_index": sample_idx, "topk_label": topk_label, "rank": rank, "top1": rank == 1})
    out["correct_rank"] = {
        "count": len(rank_rows),
        "top1": int(sum(1 for r in rank_rows if r["top1"])),
        "top1_rate": float(np.mean([r["top1"] for r in rank_rows])) if rank_rows else 0.0,
        "rank_mean": float(np.mean([r["rank"] for r in rank_rows])) if rank_rows else 0.0,
        "rank_median": float(np.median([r["rank"] for r in rank_rows])) if rank_rows else 0.0,
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
        z, target_coords = load_target_latent(sample["ss_latent"])
        with np.load(resolve_path(prior_root, sample["prior_npz"])) as data:
            correct_prior = np.asarray(data["prior_coords"], dtype=np.int64)
            correct_conf = np.asarray(data["prior_conf"], dtype=np.float32) if "prior_conf" in data else np.ones((correct_prior.shape[0],), dtype=np.float32)
        rng = np.random.default_rng(args.seed + sample_idx * 1009)
        torch.manual_seed(args.seed + sample_idx * 9973)
        base_noise = torch.randn(1, flow.in_channels, int(flow.resolution), int(flow.resolution), int(flow.resolution), device=device)
        for mode in modes:
            prior_coords, prior_conf, prior_source = make_prior_mode_coords(mode, correct_prior, correct_conf, len(target_coords), samples, prior_root, sample_idx, rng)
            cond = build_condition(encoder, ss_cond, prior_coords, prior_conf, device)
            logits = sample_logits(pipeline, flow, decoder, cond, base_noise, args, device)
            target_unique = len(set(map(tuple, target_coords[:, -3:].astype(np.int32).tolist())))
            for spec in topk_specs:
                topk = target_unique if spec == "target_unique" else int(spec)
                pred = topk_coords_from_logits(logits, topk)
                metrics = sparse_overlap_metrics(pred, target_coords)
                row = {
                    "sample_order": order,
                    "sample_index": sample_idx,
                    "uid": uid,
                    "prior_mode": mode,
                    "prior_source": prior_source,
                    "prior_points": int(prior_coords.shape[0]),
                    "topk_label": spec,
                    "topk": int(topk),
                    **metrics,
                }
                rows.append(row)
                print(
                    f"[eval_sparse_inpaint] idx={sample_idx} mode={mode} topk={spec} "
                    f"iou={row['iou']:.4f} recall={row['target_recall']:.4f} "
                    f"precision={row['pred_precision']:.4f}",
                    flush=True,
                )
    report = {"args": vars(args), "summary": summarize(rows), "rows": rows}
    write_json(output_dir / "report.json", report)
    write_csv(output_dir / "report.csv", rows)
    print(f"[eval_sparse_inpaint] wrote {output_dir / 'report.json'}", flush=True)


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
