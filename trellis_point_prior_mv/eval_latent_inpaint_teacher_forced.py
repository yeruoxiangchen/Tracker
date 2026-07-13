#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch
import torch.nn.functional as F

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis_point_prior_mv.common import load_manifest, parse_indices, resolve_path, write_json  # noqa: E402
from trellis_point_prior_mv.eval_latent_inpaint_flow import (  # noqa: E402
    add_source_rows,
    build_condition,
    build_models,
    decode_logits,
    latent_region_stats,
    summarize,
)
from trellis_point_prior_mv.eval_latent_splice_sanity import latent_mask_from_prior, normalize_latent_mask, parse_decode_specs  # noqa: E402
from trellis_point_prior_mv.eval_sparse_vae_sanity import coords4  # noqa: E402
from trellis_point_prior_mv.latent_inpaint_image_condition import SourceImageResolver  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_choices(text: str) -> list[float]:
    out: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            value = float(part)
            if not (0.0 < value < 1.0):
                raise ValueError(f"teacher-forced t should be in (0,1), got {value}")
            out.append(value)
    if not out:
        raise ValueError(f"empty t choices: {text!r}")
    return out


def safe_t_label(t: float) -> str:
    return f"tf_t{t:.2f}".replace(".", "p")


@torch.no_grad()
def teacher_forced_x0(
    *,
    flow,
    sampler,
    cond: torch.Tensor,
    q_splice: torch.Tensor,
    q_vis: torch.Tensor,
    mask: torch.Tensor,
    t: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    noise = torch.randn_like(q_splice)
    x_t, gt_v = sampler._get_model_gt(q_splice, float(t), noise)
    t_tensor = torch.tensor([1000.0 * float(t)] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
    pred_v = flow(x_t, t_tensor, cond)
    pred_x0 = sampler._pred_to_xstart(x_t, float(t), pred_v).float()
    hard_x0 = pred_x0 * (1.0 - mask) + q_vis * mask
    stats = {
        "teacher_t": float(t),
        "teacher_pred_v_mse": float(torch.nan_to_num(F.mse_loss(pred_v, gt_v), nan=0.0, posinf=1e4, neginf=1e4).detach().cpu()),
        "teacher_raw_x0_vs_q_splice_l1": float(torch.mean(torch.abs(pred_x0 - q_splice)).detach().cpu()),
        "teacher_hard_x0_vs_q_splice_l1": float(torch.mean(torch.abs(hard_x0 - q_splice)).detach().cpu()),
    }
    return hard_x0, stats


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(int(args.seed))
    payload, samples = load_manifest(args.manifest)
    indices = parse_indices(args.indices, len(samples))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    topk_specs = parse_decode_specs(args.topk)
    t_values = parse_float_choices(args.t_values)
    pipeline, flow, decoder, ss_cond = build_models(args, device)
    image_resolver = SourceImageResolver(payload.get("source_manifest")) if args.use_image_cond else None

    rows: list[dict[str, Any]] = []
    for order, sample_idx in enumerate(indices):
        sample = samples[sample_idx]
        uid = str(sample.get("uid", sample_idx))
        latent_path = resolve_path(payload.get("latent_root"), sample["latent_npz"])
        with np.load(latent_path) as data:
            q_gt = np.asarray(data["q_gt"], dtype=np.float32)
            q_vis = np.asarray(data["q_vis"], dtype=np.float32)
            saved_m_s = normalize_latent_mask(
                np.asarray(data["m_s"], dtype=np.float32),
                latent_resolution=int(args.latent_grid_resolution),
            )
            prior_coords = np.asarray(data["prior_coords"], dtype=np.int32)
            target_coords = np.asarray(data["target_coords"], dtype=np.int32)
        if args.mask_dilate64 == 0 and args.mask_dilate16 == 0:
            mask = saved_m_s
        else:
            mask = latent_mask_from_prior(
                prior_coords,
                mask_dilate64=int(args.mask_dilate64),
                mask_dilate16=int(args.mask_dilate16),
                source_resolution=int(args.source_grid_resolution),
                latent_resolution=int(args.latent_grid_resolution),
            )
        q_splice = q_gt * (1.0 - mask) + q_vis * mask
        sample_dir = output_dir / f"{sample_idx:04d}_{uid[:12]}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[latent_teacher_forced] sample={sample_idx} uid={uid} "
            f"mask={int(mask.sum())} prior={coords4(prior_coords).shape[0]} target={coords4(target_coords).shape[0]}",
            flush=True,
        )

        q_gt_t = torch.from_numpy(q_gt[None]).to(device=device, dtype=torch.float32)
        q_vis_t = torch.from_numpy(q_vis[None]).to(device=device, dtype=torch.float32)
        mask_t = torch.from_numpy(mask[None]).to(device=device, dtype=torch.float32)
        q_splice_t = torch.from_numpy(q_splice[None]).to(device=device, dtype=torch.float32)
        if image_resolver is not None:
            image_paths, image_mask_paths = image_resolver.image_mask_paths(
                sample,
                max_views=int(args.image_max_views),
                frame_select=str(args.image_frame_select),
                seed=int(args.image_select_seed),
            )
            if not args.image_use_source_mask:
                image_mask_paths = [None for _ in image_paths]
        else:
            image_paths, image_mask_paths = [], []
        cond = build_condition(
            ss_cond,
            q_vis_t,
            mask_t,
            cond_use_full_q_vis=bool(args.cond_use_full_q_vis),
            pipeline=pipeline,
            image_paths=image_paths,
            image_mask_paths=image_mask_paths,
            image_cond_aggregation=args.image_cond_aggregation,
            image_preprocess=bool(args.image_preprocess),
            image_use_source_mask=bool(args.image_use_source_mask),
            image_mask_crop_resolution=int(args.image_mask_crop_resolution),
            cond_fusion=args.cond_fusion,
        )

        base_row = {
            "sample_order": int(order),
            "sample_index": int(sample_idx),
            "uid": uid,
            "mask_dilate64": int(args.mask_dilate64),
            "mask_dilate16": int(args.mask_dilate16),
            "mask_cell_count": int(mask.sum()),
            "mask_cell_ratio": float(mask.mean()),
            "saved_m_s_vs_recomputed_mask_l1": float(np.abs(saved_m_s - mask).mean()),
            "prior_unique": int(coords4(prior_coords).shape[0]),
            "target_unique": int(coords4(target_coords).shape[0]),
            "use_image_cond": int(bool(args.use_image_cond)),
            "image_cond_view_count": int(len(image_paths)),
            "image_cond_mask_count": int(sum(1 for p in image_mask_paths if p)),
            "image_cond_aggregation": args.image_cond_aggregation,
            "image_use_source_mask": int(bool(args.image_use_source_mask)),
            "image_mask_crop_resolution": int(args.image_mask_crop_resolution),
            "cond_fusion": args.cond_fusion,
            "cond_token_count": int(cond.shape[1]),
        }
        for source, latent in {"q_gt": q_gt, "q_vis": q_vis, "q_splice": q_splice}.items():
            logits = decode_logits(decoder, torch.from_numpy(latent[None]).to(device=device, dtype=torch.float32))
            add_source_rows(
                rows=rows,
                source=source,
                latent_np=latent,
                logits=logits,
                sample_dir=sample_dir,
                base_row=base_row,
                q_gt=q_gt,
                q_vis=q_vis,
                mask=mask,
                target_coords=target_coords,
                prior_coords=prior_coords,
                topk_specs=topk_specs,
                threshold=float(args.threshold),
                prior_radius=float(args.prior_radius),
            )

        for t in t_values:
            hard_x0, teacher_stats = teacher_forced_x0(
                flow=flow,
                sampler=pipeline.sparse_structure_sampler,
                cond=cond,
                q_splice=q_splice_t,
                q_vis=q_vis_t,
                mask=mask_t,
                t=float(t),
            )
            latent_np = hard_x0.detach().cpu().numpy()[0].astype(np.float32)
            source = safe_t_label(float(t))
            logits = decode_logits(decoder, hard_x0)
            source_dir = sample_dir / source
            source_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(source_dir / "latent.npz", latent=latent_np, mask=mask.astype(np.float32))
            add_source_rows(
                rows=rows,
                source=source,
                latent_np=latent_np,
                logits=logits,
                sample_dir=sample_dir,
                base_row={
                    **base_row,
                    **teacher_stats,
                    **latent_region_stats(source, latent_np, q_gt, q_vis, mask),
                },
                q_gt=q_gt,
                q_vis=q_vis,
                mask=mask,
                target_coords=target_coords,
                prior_coords=prior_coords,
                topk_specs=topk_specs,
                threshold=float(args.threshold),
                prior_radius=float(args.prior_radius),
            )

    report = {"args": vars(args), "rows": rows, "summary": summarize(rows)}
    write_json(output_dir / "report.json", report)
    write_csv(output_dir / "report.csv", rows)
    print(f"[latent_teacher_forced] wrote {output_dir / 'report.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-forced x0 eval for latent inpainting flow.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--indices", default="0-31")
    parser.add_argument("--mask_dilate64", type=int, default=0)
    parser.add_argument("--mask_dilate16", type=int, default=0)
    parser.add_argument("--source_grid_resolution", type=int, default=64)
    parser.add_argument("--latent_grid_resolution", type=int, default=16)
    parser.add_argument("--t_values", default="0.25,0.5,0.75")
    parser.add_argument("--topk", default="4096,8192,target_unique")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--prior_radius", type=float, default=4.0)
    parser.add_argument("--cond_use_full_q_vis", action="store_true")
    parser.add_argument("--use_image_cond", action="store_true")
    parser.add_argument("--image_max_views", type=int, default=4)
    parser.add_argument("--image_frame_select", default="uniform", choices=["uniform", "first", "random"])
    parser.add_argument("--image_select_seed", type=int, default=0)
    parser.add_argument("--image_cond_aggregation", default="mean", choices=["mean", "first", "concat"])
    parser.add_argument("--image_preprocess", action="store_true")
    parser.add_argument("--image_use_source_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_mask_crop_resolution", type=int, default=518)
    parser.add_argument("--cond_fusion", default="concat", choices=["concat", "point_only", "image_only"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--latent_channels", type=int, default=8)
    parser.add_argument("--cond_channels", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    main()
