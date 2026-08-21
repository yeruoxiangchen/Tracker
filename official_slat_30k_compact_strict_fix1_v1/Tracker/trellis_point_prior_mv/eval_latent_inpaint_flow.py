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

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ar_pose_trellis.pipeline import apply_lora_to_ss_flow  # noqa: E402
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline  # noqa: E402

from trellis_point_prior_mv.common import SparsePointPriorCond, load_manifest, parse_indices, resolve_path, sparse_overlap_metrics, write_json  # noqa: E402
from trellis_point_prior_mv.eval_latent_splice_sanity import (  # noqa: E402
    add_decode_rows,
    latent_mask_from_prior,
    normalize_latent_mask,
    parse_decode_specs,
)
from trellis_point_prior_mv.eval_sparse_vae_sanity import coords4  # noqa: E402
from trellis_point_prior_mv.latent_inpaint_image_condition import (  # noqa: E402
    SourceImageResolver,
    encode_image_condition,
    fuse_point_image_condition,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def weighted_np_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    weight = np.asarray(mask, dtype=np.float32)
    if weight.ndim == 5 and weight.shape[0] == 1:
        weight = weight[0]
    if weight.ndim == 3:
        weight = weight[None, :, :, :]
    if weight.ndim != 4:
        raise ValueError(f"expected mask [1,D,H,W] or [D,H,W], got {weight.shape}")
    if weight.shape[0] == 1:
        weight = np.broadcast_to(weight, a.shape)
    denom = float(weight.sum())
    if denom <= 1e-8:
        return 0.0
    return float((np.abs(a - b) * weight).sum() / denom)


def latent_region_stats(prefix: str, latent: np.ndarray, q_gt: np.ndarray, q_vis: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    known = np.asarray(mask, dtype=np.float32)
    if known.ndim == 4:
        known = known[None, :, :, :, :]
    unknown = 1.0 - known
    return {
        f"{prefix}_vs_q_gt_l1": float(np.abs(latent - q_gt).mean()),
        f"{prefix}_vs_q_vis_l1": float(np.abs(latent - q_vis).mean()),
        f"{prefix}_known_vs_q_gt_l1": weighted_np_l1(latent, q_gt, known),
        f"{prefix}_known_vs_q_vis_l1": weighted_np_l1(latent, q_vis, known),
        f"{prefix}_unknown_vs_q_gt_l1": weighted_np_l1(latent, q_gt, unknown),
        f"{prefix}_unknown_vs_q_vis_l1": weighted_np_l1(latent, q_vis, unknown),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows), "by_source_decode": {}}
    numeric = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    keys = sorted({(row.get("source"), row.get("decode")) for row in rows})
    for source, decode in keys:
        rr = [row for row in rows if row.get("source") == source and row.get("decode") == decode]
        name = f"{source}/{decode}"
        out["by_source_decode"][name] = {"count": len(rr)}
        for key in numeric:
            vals = [float(r[key]) for r in rr if isinstance(r.get(key), (int, float))]
            if vals:
                out["by_source_decode"][name][f"{key}_mean"] = float(np.mean(vals))
                out["by_source_decode"][name][f"{key}_median"] = float(np.median(vals))
    return out


def build_models(args: argparse.Namespace, device: torch.device):
    print(f"[eval_latent_inpaint_flow] loading pipeline weights={args.weights}", flush=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.to(device)
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    for p in flow.parameters():
        p.requires_grad = False
    if not args.no_lora:
        flow = apply_lora_to_ss_flow(flow, r=args.lora_rank, alpha=args.lora_alpha).to(device).eval()
    decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    for p in decoder.parameters():
        p.requires_grad = False
    ss_cond = SparsePointPriorCond(
        latent_channels=args.latent_channels,
        cond_channels=args.cond_channels,
        grid_resolution=args.latent_grid_resolution,
    ).to(device).eval()
    if args.checkpoint:
        if args.no_lora:
            print(
                "[eval_latent_inpaint_flow][WARN] --checkpoint was provided with --no_lora; "
                "loading only ss_cond if present and ignoring ss_flow_model LoRA keys.",
                flush=True,
            )
        state = torch.load(args.checkpoint, map_location="cpu")
        state = state.get("state_dict", state)
        flow_state = {k.replace("ss_flow_model.", "", 1): v for k, v in state.items() if k.startswith("ss_flow_model.")}
        cond_state = {k.replace("ss_cond.", "", 1): v for k, v in state.items() if k.startswith("ss_cond.")}
        if flow_state and not args.no_lora:
            missing, unexpected = flow.load_state_dict(flow_state, strict=False)
            print(f"[eval_latent_inpaint_flow] flow load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if cond_state:
            missing, unexpected = ss_cond.load_state_dict(cond_state, strict=False)
            print(f"[eval_latent_inpaint_flow] cond load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return pipeline, flow, decoder, ss_cond


def decode_logits(decoder, latent: torch.Tensor) -> torch.Tensor:
    logits = decoder(latent.to(dtype=next(decoder.parameters()).dtype)).float()
    if logits.shape[1] != 1:
        logits = logits.max(dim=1, keepdim=True).values
    return logits


def build_condition(
    ss_cond: SparsePointPriorCond,
    q_vis: torch.Tensor,
    mask: torch.Tensor,
    *,
    cond_use_full_q_vis: bool,
    pipeline=None,
    image_paths: list[str] | None = None,
    image_mask_paths: list[str | None] | None = None,
    image_cond_aggregation: str = "mean",
    image_preprocess: bool = False,
    image_use_source_mask: bool = True,
    image_mask_crop_resolution: int = 518,
    cond_fusion: str = "concat",
) -> torch.Tensor:
    confidence = mask.clamp(0.0, 1.0)
    cond_latent = q_vis if cond_use_full_q_vis else q_vis * mask
    point_cond = ss_cond(cond_latent, mask, confidence)
    image_cond = None
    if image_paths:
        if pipeline is None:
            raise ValueError("image_paths were provided but pipeline is None")
        image_cond = encode_image_condition(
            pipeline,
            image_paths,
            device=point_cond.device,
            dtype=point_cond.dtype,
            aggregation=image_cond_aggregation,
            preprocess=image_preprocess,
            mask_paths=image_mask_paths,
            use_source_mask=image_use_source_mask,
            mask_crop_resolution=image_mask_crop_resolution,
        )
    return fuse_point_image_condition(point_cond, image_cond, cond_fusion)


def sampler_runtime_config(pipeline, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, float | int | None]]:
    sampler_params = {**pipeline.sparse_structure_sampler_params}
    native_steps = int(sampler_params.pop("steps", 12))
    sampler_params.pop("verbose", None)
    native_rescale_t = float(sampler_params.pop("rescale_t", 1.0))
    native_cfg_strength = float(sampler_params.pop("cfg_strength", 1.0))
    native_guidance_rescale = sampler_params.pop("guidance_rescale", None)
    effective_steps = int(native_steps if int(args.steps) <= 0 else int(args.steps))
    effective_cfg_strength = (
        native_cfg_strength
        if bool(args.use_native_ss_cfg) or args.guidance_strength is None
        else float(args.guidance_strength)
    )
    effective_guidance_rescale = (
        native_guidance_rescale
        if bool(args.use_native_ss_cfg) or args.guidance_rescale is None
        else float(args.guidance_rescale)
    )
    config = {
        "remaining_params": sampler_params,
        "native_steps": native_steps,
        "native_rescale_t": native_rescale_t,
        "native_cfg_strength": native_cfg_strength,
        "native_guidance_rescale": native_guidance_rescale,
        "effective_steps": effective_steps,
        "effective_rescale_t": native_rescale_t,
        "effective_cfg_strength": effective_cfg_strength,
        "effective_guidance_rescale": effective_guidance_rescale,
    }
    row = {
        "native_ss_steps": int(native_steps),
        "native_ss_cfg_strength": float(native_cfg_strength),
        "native_ss_rescale_t": float(native_rescale_t),
        "native_ss_guidance_rescale": None if native_guidance_rescale is None else float(native_guidance_rescale),
        "effective_ss_steps": int(effective_steps),
        "effective_ss_cfg_strength": float(effective_cfg_strength),
        "effective_ss_guidance_rescale": None if effective_guidance_rescale is None else float(effective_guidance_rescale),
        "use_native_ss_cfg": int(bool(args.use_native_ss_cfg)),
        "no_lora": int(bool(args.no_lora)),
    }
    return config, row


@torch.no_grad()
def sample_latent(
    pipeline,
    flow,
    cond: torch.Tensor,
    q_vis: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    sampler = pipeline.sparse_structure_sampler
    runtime, _ = sampler_runtime_config(pipeline, args)
    rescale_t = float(runtime["effective_rescale_t"])
    model_kwargs = dict(runtime["remaining_params"])
    model_kwargs.update({"neg_cond": torch.zeros_like(cond), "cfg_strength": float(runtime["effective_cfg_strength"])})
    if runtime["effective_guidance_rescale"] is not None:
        model_kwargs["guidance_rescale"] = float(runtime["effective_guidance_rescale"])

    noise = initial_noise
    if noise is None:
        noise = torch.randn(
            1,
            flow.in_channels,
            int(flow.resolution),
            int(flow.resolution),
            int(flow.resolution),
            device=device,
        )
    sample = noise.clone()
    clamp = (mask * float(args.known_latent_clamp_strength)).clamp(0.0, 1.0)
    clamp_start_t = float(args.known_clamp_start_t)
    known_noise = noise.clone()
    if args.known_latent_clamp_strength > 0 and args.clamp_initial_noise and 1.0 <= clamp_start_t + 1e-8:
        known_xt = sampler._xstart_to_x_t(q_vis, 1.0, known_noise)
        sample = sample * (1.0 - clamp) + known_xt * clamp

    steps = int(runtime["effective_steps"])
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    for t, t_prev in list((t_seq[i], t_seq[i + 1]) for i in range(steps)):
        out = sampler.sample_once(flow, sample, float(t), float(t_prev), cond, **model_kwargs)
        sample = out.pred_x_prev
        if args.known_latent_clamp_strength > 0 and float(t_prev) <= clamp_start_t + 1e-8:
            known_xt_prev = sampler._xstart_to_x_t(q_vis, float(t_prev), known_noise)
            sample = sample * (1.0 - clamp) + known_xt_prev * clamp
    return sample.float()


@torch.no_grad()
def sample_latent_native_sampler(
    pipeline,
    flow,
    cond: torch.Tensor,
    args: argparse.Namespace,
    initial_noise: torch.Tensor,
) -> torch.Tensor:
    runtime, _ = sampler_runtime_config(pipeline, args)
    sampler_params = dict(runtime["remaining_params"])
    sampler_params.update(
        {
            "steps": int(runtime["effective_steps"]),
            "rescale_t": float(runtime["effective_rescale_t"]),
            "cfg_strength": float(runtime["effective_cfg_strength"]),
            "verbose": False,
        }
    )
    if runtime["effective_guidance_rescale"] is not None:
        sampler_params["guidance_rescale"] = float(runtime["effective_guidance_rescale"])
    out = pipeline.sparse_structure_sampler.sample(
        flow,
        initial_noise.clone(),
        cond=cond,
        neg_cond=torch.zeros_like(cond),
        **sampler_params,
    )
    return out.samples.float()


def add_source_rows(
    *,
    rows: list[dict[str, Any]],
    source: str,
    latent_np: np.ndarray,
    logits: torch.Tensor,
    sample_dir: Path,
    base_row: dict[str, Any],
    q_gt: np.ndarray,
    q_vis: np.ndarray,
    mask: np.ndarray,
    target_coords: np.ndarray,
    prior_coords: np.ndarray,
    topk_specs: list[str],
    threshold: float,
    prior_radius: float,
) -> None:
    source_dir = sample_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(source_dir / "latent.npz", latent=latent_np.astype(np.float32), mask=mask.astype(np.float32))
    input_coords = target_coords if source != "q_vis" else prior_coords
    add_decode_rows(
        rows=rows,
        base_row={
            **base_row,
            "source": source,
            **latent_region_stats(source, latent_np, q_gt, q_vis, mask),
        },
        logits=logits,
        source_dir=source_dir,
        target_coords=target_coords,
        prior_coords=prior_coords,
        input_coords=input_coords,
        topk_specs=topk_specs,
        threshold=float(threshold),
        target_unique=int(coords4(target_coords).shape[0]),
        prior_radius=float(prior_radius),
    )


def main() -> None:
    args = parse_args()
    if not args.checkpoint and str(args.cond_fusion).lower() not in {"image_only", "image"}:
        raise ValueError("--checkpoint is required unless --cond_fusion image_only is used")
    if args.no_lora and args.checkpoint and str(args.cond_fusion).lower() not in {"image_only", "image"}:
        raise ValueError("--no_lora with a checkpoint is only meaningful for --cond_fusion image_only")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(int(args.seed))
    payload, samples = load_manifest(args.manifest)
    indices = parse_indices(args.indices, len(samples))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    topk_specs = parse_decode_specs(args.topk)
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
            f"[eval_latent_inpaint_flow] sample={sample_idx} uid={uid} "
            f"mask={int(mask.sum())} prior={coords4(prior_coords).shape[0]} target={coords4(target_coords).shape[0]}",
            flush=True,
        )

        q_vis_t = torch.from_numpy(q_vis[None]).to(device=device, dtype=torch.float32)
        mask_t = torch.from_numpy(mask[None]).to(device=device, dtype=torch.float32)
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
        initial_noise = torch.randn(
            1,
            flow.in_channels,
            int(flow.resolution),
            int(flow.resolution),
            int(flow.resolution),
            device=device,
        )
        pred_t = sample_latent(pipeline, flow, cond, q_vis_t, mask_t, args, device, initial_noise=initial_noise)
        pred_np = pred_t.detach().cpu().numpy()[0].astype(np.float32)
        native_pred_np = None
        if args.add_native_sampler_pred:
            native_pred_t = sample_latent_native_sampler(pipeline, flow, cond, args, initial_noise=initial_noise)
            native_pred_np = native_pred_t.detach().cpu().numpy()[0].astype(np.float32)

        _, sampler_row = sampler_runtime_config(pipeline, args)
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
            **sampler_row,
        }
        sources = {
            "q_gt": q_gt,
            "q_vis": q_vis,
            "q_splice": q_splice,
            "q_pred": pred_np,
        }
        if native_pred_np is not None:
            sources["q_pred_native_sampler"] = native_pred_np
        for source, latent in sources.items():
            latent_t = torch.from_numpy(latent[None]).to(device=device, dtype=torch.float32)
            logits = decode_logits(decoder, latent_t)
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

    report = {"args": vars(args), "rows": rows, "summary": summarize(rows)}
    write_json(output_dir / "report.json", report)
    write_csv(output_dir / "report.csv", rows)
    print(f"[eval_latent_inpaint_flow] wrote {output_dir / 'report.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latent inpainting flow before mesh/slat.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--indices", default="0-7")
    parser.add_argument("--mask_dilate64", type=int, default=0)
    parser.add_argument("--mask_dilate16", type=int, default=0)
    parser.add_argument("--source_grid_resolution", type=int, default=64)
    parser.add_argument("--latent_grid_resolution", type=int, default=16)
    parser.add_argument("--topk", default="4096,8192,target_unique")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--prior_radius", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=12, help="Use <=0 to reuse TRELLIS native sparse sampler steps.")
    parser.add_argument("--guidance_strength", type=float, default=None)
    parser.add_argument("--guidance_rescale", type=float, default=None)
    parser.add_argument("--use_native_ss_cfg", action="store_true")
    parser.add_argument("--add_native_sampler_pred", action="store_true")
    parser.add_argument("--known_latent_clamp_strength", type=float, default=1.0)
    parser.add_argument("--known_clamp_start_t", type=float, default=0.5)
    parser.add_argument("--clamp_initial_noise", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--latent_channels", type=int, default=8)
    parser.add_argument("--cond_channels", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    main()
