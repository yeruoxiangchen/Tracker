#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline  # noqa: E402

from points_to_3d_strict.strict_core import (  # noqa: E402
    PROJECTION_AWARE_AGGREGATIONS,
    StrictLatentDataset,
    blockwise_global_modulation_input_dim,
    blockwise_token_modulation_input_dim,
    build_blockwise_global_modulation_context,
    build_blockwise_token_modulation_context,
    build_condition_adapter,
    build_q_comb_t,
    build_strict_input_projection_features,
    encode_multiview_condition,
    inject_flow_blockwise_global_modulation,
    inject_flow_blockwise_token_modulation,
    inject_flow_lora,
    load_condition_adapter_checkpoint,
    load_strict_checkpoint,
    replace_sparse_flow_input_layer,
    sample_stock_sparse_latent,
    sample_points_to_3d_strict,
    save_json,
    set_flow_blockwise_global_modulation_context,
    set_flow_blockwise_token_modulation_context,
    strict_collate,
    strict_input_projection_channel_count,
)
from trellis_point_prior_mv.eval_latent_splice_sanity import add_decode_rows, parse_decode_specs  # noqa: E402
from trellis_point_prior_mv.eval_sparse_vae_sanity import coords4  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows), "by_source_decode": {}}
    keys = sorted({(row.get("source"), row.get("decode")) for row in rows})
    numeric = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float))})
    for source, decode in keys:
        rr = [r for r in rows if r.get("source") == source and r.get("decode") == decode]
        name = f"{source}/{decode}"
        out["by_source_decode"][name] = {"count": len(rr)}
        for key in numeric:
            vals = [float(r[key]) for r in rr if isinstance(r.get(key), (int, float))]
            if vals:
                out["by_source_decode"][name][f"{key}_mean"] = float(np.mean(vals))
                out["by_source_decode"][name][f"{key}_median"] = float(np.median(vals))
    return out


@torch.no_grad()
def decode_logits(decoder, latent: torch.Tensor) -> torch.Tensor:
    logits = decoder(latent.to(dtype=next(decoder.parameters()).dtype)).float()
    if logits.shape[1] != 1:
        logits = logits.max(dim=1, keepdim=True).values
    return logits


@torch.no_grad()
def encode_cond_for_batch(
    pipeline,
    batch: dict,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    adapter: torch.nn.Module | None = None,
) -> torch.Tensor:
    conds = []
    q_vis = batch["q_vis"].to(device=device, dtype=torch.float32)
    mask = batch["m_s"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    view_pose_all = batch.get("view_pose_features")
    if view_pose_all is not None:
        view_pose_all = view_pose_all.to(device=device, dtype=torch.float32)
    view_intrinsics_all = batch.get("view_intrinsics")
    view_extrinsics_all = batch.get("view_extrinsics")
    view_camera_forward_all = batch.get("view_camera_forward_sign")
    for row_idx, (image_paths, mask_paths) in enumerate(zip(batch["image_paths"], batch["image_mask_paths"])):
        view_pose = view_pose_all[row_idx : row_idx + 1] if view_pose_all is not None and view_pose_all.numel() > 0 else None
        projection_build_kwargs = None
        if str(args.image_cond_aggregation).lower() in PROJECTION_AWARE_AGGREGATIONS:
            if view_intrinsics_all is None or view_extrinsics_all is None:
                raise ValueError("projection-aware condition requires view_intrinsics/view_extrinsics in batch")
            projection_build_kwargs = {
                "mask_paths": list(mask_paths),
                "intrinsics": view_intrinsics_all[row_idx].detach().cpu().numpy(),
                "extrinsics": view_extrinsics_all[row_idx].detach().cpu().numpy(),
                "extrinsics_type": batch.get("view_extrinsics_type", ["c2w"])[row_idx],
                "camera_forward_sign": float(view_camera_forward_all[row_idx].item()) if view_camera_forward_all is not None else 1.0,
                "grid_transform": batch.get("projection_grid_transform", ["pixal3d_rotation"])[row_idx],
                "latent_resolution": int(mask.shape[-1]),
                "image_mask_crop_resolution": int(args.image_mask_crop_resolution),
                "device": device,
                "dtype": torch.float32,
            }
        conds.append(
            encode_multiview_condition(
                pipeline,
                list(image_paths),
                list(mask_paths),
                device=device,
                dtype=dtype,
                aggregation=args.image_cond_aggregation,
                image_use_source_mask=bool(args.image_use_source_mask),
                image_mask_crop_resolution=int(args.image_mask_crop_resolution),
                adapter=adapter,
                q_vis=q_vis[row_idx : row_idx + 1],
                mask=mask[row_idx : row_idx + 1],
                view_pose=view_pose,
                projection_build_kwargs=projection_build_kwargs,
                projection_token_max_cells=int(args.condition_projection_token_max_cells),
            )
        )
    token_counts = {int(c.shape[1]) for c in conds}
    if len(token_counts) != 1:
        raise ValueError(f"variable multiview token counts in batch: {sorted(token_counts)}")
    return torch.cat(conds, dim=0)


def latent_l1_stats(prefix: str, latent: np.ndarray, q_gt: np.ndarray, q_vis: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    latent = np.asarray(latent, dtype=np.float32)
    q_gt = np.asarray(q_gt, dtype=np.float32)
    q_vis = np.asarray(q_vis, dtype=np.float32)
    known = np.asarray(mask, dtype=np.float32).squeeze()
    if latent.shape != q_gt.shape or latent.shape != q_vis.shape:
        raise ValueError(f"latent/q_gt/q_vis shape mismatch: {latent.shape} {q_gt.shape} {q_vis.shape}")
    if latent.ndim == 4:
        spatial_shape = latent.shape[-3:]
        if known.shape != spatial_shape:
            raise ValueError(f"mask spatial shape {known.shape} does not match latent spatial shape {spatial_shape}")
        known_b = np.broadcast_to(known[None, ...], latent.shape)
    elif latent.ndim == 5:
        spatial_shape = latent.shape[-3:]
        if known.shape != spatial_shape:
            raise ValueError(f"mask spatial shape {known.shape} does not match latent spatial shape {spatial_shape}")
        known_b = np.broadcast_to(known[None, None, ...], latent.shape)
    else:
        raise ValueError(f"expected latent shape [C,D,H,W] or [B,C,D,H,W], got {latent.shape}")
    unknown_b = 1.0 - known_b
    known_sel = known_b > 0.5
    unknown_sel = unknown_b > 0.5
    return {
        f"{prefix}_vs_q_gt_l1": float(np.abs(latent - q_gt).mean()),
        f"{prefix}_vs_q_vis_l1": float(np.abs(latent - q_vis).mean()),
        f"{prefix}_known_vs_q_vis_l1": float(np.abs(latent[known_sel] - q_vis[known_sel]).mean()) if known_sel.any() else 0.0,
        f"{prefix}_unknown_vs_q_gt_l1": float(np.abs(latent[unknown_sel] - q_gt[unknown_sel]).mean()) if unknown_sel.any() else 0.0,
    }


def load_models(args: argparse.Namespace, device: torch.device):
    print(f"[points_to_3d_strict][eval] loading TRELLIS weights={args.weights}", flush=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.to(device)
    for module in pipeline.models.values():
        module.eval()
        for p in module.parameters():
            p.requires_grad = False
    flow = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    adapter = build_condition_adapter(args, cond_dim=int(args.condition_adapter_dim))
    if adapter is not None:
        adapter = adapter.to(device).eval()
    return pipeline, flow, decoder, adapter


def needs_stock_latent(args: argparse.Namespace) -> bool:
    return str(args.init_mode) in {"stock_latent", "stock_t"} or bool(args.include_stock_source)


@torch.no_grad()
def precompute_stock_latents(
    *,
    pipeline,
    loader,
    args: argparse.Namespace,
    device: torch.device,
    adapter: torch.nn.Module | None = None,
) -> dict[int, torch.Tensor]:
    stock: dict[int, torch.Tensor] = {}
    for order, batch in enumerate(loader):
        cond = encode_cond_for_batch(
            pipeline,
            batch,
            args,
            device,
            dtype=next(pipeline.models["sparse_structure_flow_model"].parameters()).dtype,
            adapter=adapter,
        )
        latent = sample_stock_sparse_latent(
            pipeline=pipeline,
            cond=cond,
            steps=None if int(args.stock_steps) <= 0 else int(args.stock_steps),
            cfg_strength=None if float(args.stock_cfg_strength) <= 0 else float(args.stock_cfg_strength),
            rescale_t=None if float(args.stock_rescale_t) <= 0 else float(args.stock_rescale_t),
            seed=int(args.seed + order * 9173),
        )
        stock[order] = latent.detach().cpu()
        print(
            f"[points_to_3d_strict][eval] stock latent sample={order} "
            f"shape={tuple(latent.shape)}",
            flush=True,
        )
    return stock


def effective_start_t(args: argparse.Namespace) -> float:
    start_t = float(args.start_t)
    rescale_t = float(args.rescale_t)
    return float(rescale_t * start_t / (1.0 + (rescale_t - 1.0) * start_t))


def build_initial_sample(
    *,
    args: argparse.Namespace,
    sampler,
    q_vis: torch.Tensor,
    q_splice: torch.Tensor,
    q_stock: torch.Tensor | None,
) -> torch.Tensor:
    mode = str(args.init_mode)
    noise = torch.randn_like(q_vis)
    if mode == "noise":
        return noise
    if mode == "q_splice":
        return q_splice
    if mode == "q_splice_noise":
        return q_splice + float(args.q_splice_noise_scale) * noise
    if mode == "q_splice_t":
        return sampler._xstart_to_x_t(q_splice, effective_start_t(args), noise).float()
    if mode == "stock_latent":
        if q_stock is None:
            raise ValueError("init_mode=stock_latent requires precomputed q_stock")
        return q_stock.to(device=q_vis.device, dtype=q_vis.dtype)
    if mode == "stock_t":
        if q_stock is None:
            raise ValueError("init_mode=stock_t requires precomputed q_stock")
        return sampler._xstart_to_x_t(q_stock.to(device=q_vis.device, dtype=q_vis.dtype), effective_start_t(args), noise).float()
    raise ValueError(f"unsupported init_mode={mode}")


def add_latent_source(
    *,
    rows: list[dict[str, Any]],
    source: str,
    latent_np: np.ndarray,
    decoder,
    device: torch.device,
    sample_dir: Path,
    base_row: dict[str, Any],
    target_coords: np.ndarray,
    prior_coords: np.ndarray,
    input_coords: np.ndarray,
    topk_specs: list[str],
    threshold: float,
    prior_radius: float,
) -> None:
    source_dir = sample_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(source_dir / "latent.npz", latent=latent_np.astype(np.float32))
    latent_t = torch.from_numpy(latent_np[None]).to(device=device, dtype=torch.float32)
    logits = decode_logits(decoder, latent_t)
    add_decode_rows(
        rows=rows,
        base_row={**base_row, "source": source},
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
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    pipeline, flow, decoder, adapter = load_models(args, device)
    dataset = StrictLatentDataset(
        args.manifest,
        indices=args.indices,
        mask_dilate64=args.mask_dilate64,
        mask_dilate16=args.mask_dilate16,
        use_image_cond=True,
        image_max_views=args.image_max_views,
        image_frame_select=args.image_frame_select,
        image_select_seed=args.image_select_seed,
        image_use_source_mask=args.image_use_source_mask,
        projection_grid_transform=args.projection_grid_transform,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=strict_collate)
    total_samples = len(dataset)
    print(
        f"[points_to_3d_strict][eval] samples={total_samples} indices={args.indices} "
        f"init={args.init_mode} start_t={args.start_t} steps={args.steps}",
        file=sys.stderr,
        flush=True,
    )
    topk_specs = parse_decode_specs(args.topk)
    sampler = pipeline.sparse_structure_sampler
    stock_latents = precompute_stock_latents(pipeline=pipeline, loader=loader, args=args, device=device, adapter=adapter) if needs_stock_latent(args) else {}
    extra_input_channels = strict_input_projection_channel_count(args)
    replace_sparse_flow_input_layer(flow, mask_channels=1 + extra_input_channels)
    blockwise_summary = {"enabled": False}
    if args.blockwise_global_modulation:
        blockwise_summary = inject_flow_blockwise_global_modulation(
            flow,
            input_dim=blockwise_global_modulation_input_dim(args),
            hidden_dim=int(args.blockwise_global_hidden_dim),
            target_blocks=int(args.blockwise_global_target_blocks),
            mode=str(args.blockwise_global_mode),
        )
        print(
            f"[points_to_3d_strict][eval] blockwise_global label={blockwise_summary.get('label')} "
            f"mode={blockwise_summary.get('mode')} input_dim={blockwise_summary.get('input_dim')} "
            f"hidden={blockwise_summary.get('hidden_dim')}",
            flush=True,
        )
    blockwise_token_summary = {"enabled": False}
    if args.blockwise_token_modulation:
        blockwise_token_summary = inject_flow_blockwise_token_modulation(
            flow,
            input_dim=blockwise_token_modulation_input_dim(args, patch_size=int(flow.patch_size)),
            hidden_dim=int(args.blockwise_token_hidden_dim),
            target_blocks=int(args.blockwise_token_target_blocks),
            mode=str(args.blockwise_token_mode),
        )
        print(
            f"[points_to_3d_strict][eval] blockwise_token label={blockwise_token_summary.get('label')} "
            f"mode={blockwise_token_summary.get('mode')} input_dim={blockwise_token_summary.get('input_dim')} "
            f"hidden={blockwise_token_summary.get('hidden_dim')}",
            flush=True,
        )
    if args.flow_lora:
        summary = inject_flow_lora(
            flow,
            rank=int(args.lora_rank),
            alpha=float(args.lora_alpha),
            target_blocks=int(args.lora_target_blocks),
        )
        print(
            f"[points_to_3d_strict][eval] flow_lora label={summary.get('label')} "
            f"rank={args.lora_rank} alpha={args.lora_alpha} target_blocks={args.lora_target_blocks}",
            flush=True,
        )
    load_strict_checkpoint(flow, args.checkpoint)
    load_condition_adapter_checkpoint(adapter, args.checkpoint)
    flow.eval()
    if adapter is not None:
        adapter.eval()
    rows: list[dict[str, Any]] = []
    eval_start_wall = time.time()
    for order, batch in enumerate(loader):
        sample_start_wall = time.time()
        uid = batch["uids"][0]
        q_gt = batch["q_gt"].to(device=device, dtype=torch.float32)
        q_vis = batch["q_vis"].to(device=device, dtype=torch.float32)
        mask = batch["m_s"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        extra_inputs = build_strict_input_projection_features(
            batch=batch,
            args=args,
            mask=mask,
            device=device,
            dtype=torch.float32,
        )
        if args.blockwise_global_modulation:
            blockwise_context = build_blockwise_global_modulation_context(
                q_vis=q_vis,
                mask=mask,
                extra_inputs=extra_inputs,
                projection_channels=extra_input_channels,
            )
            set_flow_blockwise_global_modulation_context(flow, blockwise_context)
        if args.blockwise_token_modulation:
            blockwise_token_context = build_blockwise_token_modulation_context(
                q_vis=q_vis,
                mask=mask,
                extra_inputs=extra_inputs,
                projection_channels=extra_input_channels,
                patch_size=int(flow.patch_size),
            )
            set_flow_blockwise_token_modulation_context(flow, blockwise_token_context)
        cond = encode_cond_for_batch(pipeline, batch, args, device, dtype=next(flow.parameters()).dtype, adapter=adapter)
        q_gt_np = q_gt.detach().cpu().numpy()[0].astype(np.float32)
        q_vis_np = q_vis.detach().cpu().numpy()[0].astype(np.float32)
        mask_np = mask.detach().cpu().numpy()[0].astype(np.float32)
        q_splice = mask * q_vis + (1.0 - mask) * q_gt
        q_splice_np = q_splice.detach().cpu().numpy()[0].astype(np.float32)
        q_stock = stock_latents.get(order)
        q_stock_t = q_stock.to(device=device, dtype=torch.float32) if q_stock is not None else None
        q_stock_np = q_stock_t.detach().cpu().numpy()[0].astype(np.float32) if q_stock_t is not None else None
        q_stock_clamp = mask * q_vis + (1.0 - mask) * q_stock_t if q_stock_t is not None else None
        q_stock_clamp_np = q_stock_clamp.detach().cpu().numpy()[0].astype(np.float32) if q_stock_clamp is not None else None
        prior_coords = batch["prior_coords"][0]
        target_coords = batch["target_coords"][0]
        sample_dir = output_dir / f"{order:04d}_{uid[:12]}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[points_to_3d_strict][eval] sample={order + 1}/{total_samples} uid={uid} "
            f"target={coords4(target_coords).shape[0]} prior={coords4(prior_coords).shape[0]} tokens={cond.shape[1]}",
            file=sys.stderr,
            flush=True,
        )
        initial_sample = build_initial_sample(
            args=args,
            sampler=sampler,
            q_vis=q_vis,
            q_splice=q_splice,
            q_stock=q_stock_t,
        )
        pred = sample_points_to_3d_strict(
            flow=flow,
            sampler=sampler,
            q_vis=q_vis,
            mask=mask,
            cond=cond,
            steps=int(args.steps),
            inpaint_steps=int(args.steps) - int(args.boundary_refine_steps),
            rescale_t=float(args.rescale_t),
            cfg_strength=float(args.cfg_strength),
            initial_sample=initial_sample,
            extra_inputs=extra_inputs,
            start_t=float(args.start_t),
        )
        pred_np = pred.detach().cpu().numpy()[0].astype(np.float32)
        base_row = {
            "sample_order": int(order),
            "uid": uid,
            "mask_cell_count": int(mask.sum().item()),
            "mask_cell_ratio": float(mask.mean().item()),
            "image_cond_view_count": int(len(batch["image_paths"][0])),
            "image_cond_aggregation": str(args.image_cond_aggregation),
            "image_frame_select": str(args.image_frame_select),
            "image_cond_token_count": int(cond.shape[1]),
            "projection_token_max_cells": int(args.condition_projection_token_max_cells),
            "strict_input_projection_grid": int(args.strict_input_projection_grid),
            "strict_input_projection_channels": int(extra_input_channels),
            "strict_input_channel_count": int(q_vis.shape[1] + 1 + extra_input_channels),
            "blockwise_global_modulation": int(args.blockwise_global_modulation),
            "blockwise_global_target_blocks": int(args.blockwise_global_target_blocks),
            "blockwise_global_hidden_dim": int(args.blockwise_global_hidden_dim),
            "blockwise_global_mode": str(args.blockwise_global_mode),
            "blockwise_token_modulation": int(args.blockwise_token_modulation),
            "blockwise_token_target_blocks": int(args.blockwise_token_target_blocks),
            "blockwise_token_hidden_dim": int(args.blockwise_token_hidden_dim),
            "blockwise_token_mode": str(args.blockwise_token_mode),
            "flow_lora": int(args.flow_lora),
            "lora_rank": int(args.lora_rank),
            "lora_alpha": float(args.lora_alpha),
            "lora_target_blocks": int(args.lora_target_blocks),
            "condition_adapter": int(adapter is not None),
            "steps": int(args.steps),
            "inpaint_steps": int(args.steps) - int(args.boundary_refine_steps),
            "boundary_refine_steps": int(args.boundary_refine_steps),
            "cfg_strength": float(args.cfg_strength),
            "init_mode": str(args.init_mode),
            "start_t": float(args.start_t),
            "effective_start_t": float(effective_start_t(args)),
            "pure_noise_probe": int(str(args.init_mode) == "noise" and abs(float(args.start_t) - 1.0) < 1.0e-6),
            "q_splice_noise_scale": float(args.q_splice_noise_scale),
            "has_stock_latent": int(q_stock_np is not None),
        }
        sources = {
            "q_gt": (q_gt_np, target_coords),
            "q_vis": (q_vis_np, prior_coords),
            "q_splice": (q_splice_np, target_coords),
            "q_pred": (pred_np, target_coords),
        }
        if q_stock_np is not None:
            sources["q_stock"] = (q_stock_np, target_coords)
        if q_stock_clamp_np is not None:
            sources["q_stock_clamp"] = (q_stock_clamp_np, target_coords)
        for source, (latent_np, input_coords) in sources.items():
            add_latent_source(
                rows=rows,
                source=source,
                latent_np=latent_np,
                decoder=decoder,
                device=device,
                sample_dir=sample_dir,
                base_row={**base_row, **latent_l1_stats(source, latent_np, q_gt_np, q_vis_np, mask_np)},
                target_coords=target_coords,
                prior_coords=prior_coords,
                input_coords=input_coords,
                topk_specs=topk_specs,
                threshold=float(args.threshold),
                prior_radius=float(args.prior_radius),
            )
        for t in parse_float_list(args.teacher_forced_t):
            noise = torch.randn_like(q_gt)
            x_t, gt_v = sampler._get_model_gt(q_gt, float(t), noise)
            x_inp, _ = build_q_comb_t(
                q_gt=q_gt,
                q_vis=q_vis,
                mask=mask,
                sampler=sampler,
                t=float(t),
                noise=noise,
                extra_inputs=extra_inputs,
            )
            t_tensor = torch.tensor([1000.0 * float(t)], device=device, dtype=torch.float32)
            pred_v = flow(x_inp, t_tensor, cond)
            pred_x0 = sampler._pred_to_xstart(x_t, float(t), pred_v).float()
            tf_raw_np = pred_x0.detach().cpu().numpy()[0].astype(np.float32)
            pred_x0_hard = mask * q_vis + (1.0 - mask) * pred_x0
            tf_hard_np = pred_x0_hard.detach().cpu().numpy()[0].astype(np.float32)
            tf_base = {
                **base_row,
                "teacher_forced_t": float(t),
                "teacher_forced_v_mse": float(torch.mean((pred_v.float() - gt_v.float()) ** 2).detach().cpu()),
            }
            add_latent_source(
                rows=rows,
                source=f"teacher_forced_raw_t{str(t).replace('.', 'p')}",
                latent_np=tf_raw_np,
                decoder=decoder,
                device=device,
                sample_dir=sample_dir,
                base_row={
                    **tf_base,
                    **latent_l1_stats("teacher_forced_raw", tf_raw_np, q_gt_np, q_vis_np, mask_np),
                },
                target_coords=target_coords,
                prior_coords=prior_coords,
                input_coords=target_coords,
                topk_specs=topk_specs,
                threshold=float(args.threshold),
                prior_radius=float(args.prior_radius),
            )
            add_latent_source(
                rows=rows,
                source=f"teacher_forced_hard_t{str(t).replace('.', 'p')}",
                latent_np=tf_hard_np,
                decoder=decoder,
                device=device,
                sample_dir=sample_dir,
                base_row={
                    **tf_base,
                    **latent_l1_stats("teacher_forced_hard", tf_hard_np, q_gt_np, q_vis_np, mask_np),
                },
                target_coords=target_coords,
                prior_coords=prior_coords,
                input_coords=target_coords,
                topk_specs=topk_specs,
                threshold=float(args.threshold),
                prior_radius=float(args.prior_radius),
            )
        done_count = order + 1
        if done_count % max(1, int(args.progress_every)) == 0 or done_count == total_samples:
            elapsed = max(1e-6, time.time() - eval_start_wall)
            samples_per_sec = float(done_count) / elapsed
            eta_sec = max(0, total_samples - done_count) / max(samples_per_sec, 1e-6)
            print(
                f"[points_to_3d_strict][eval] done={done_count}/{total_samples} uid={uid} "
                f"sample_time={time.time() - sample_start_wall:.1f}s "
                f"elapsed={elapsed/60.0:.1f}m eta={eta_sec/60.0:.1f}m",
                file=sys.stderr,
                flush=True,
            )
    report = {"args": vars(args), "rows": rows, "summary": summarize(rows)}
    save_json(output_dir / "report.json", report)
    write_csv(output_dir / "report.csv", rows)
    print(f"[points_to_3d_strict][eval] wrote {output_dir / 'report.json'}", flush=True)


def parse_float_list(text: str) -> list[float]:
    out = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate strict Points-to-3D sparse latent inpainting.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--indices", default="0")
    parser.add_argument("--mask_dilate64", type=int, default=0)
    parser.add_argument("--mask_dilate16", type=int, default=0)
    parser.add_argument("--image_max_views", type=int, default=4)
    parser.add_argument("--image_frame_select", default="uniform", choices=["uniform", "first", "random"])
    parser.add_argument("--image_select_seed", type=int, default=0)
    parser.add_argument(
        "--image_cond_aggregation",
        default="concat",
        choices=[
            "concat",
            "mean",
            "first",
            "view_gated",
            "geometry_view_gated",
            "geo_view_gated",
            "pose_view_gated",
            "projection_view_gated",
            "projection_aware",
            "projection_geo_view_gated",
            "projection_token_view_gated",
            "projection_token_aware",
            "projection_v2",
        ],
    )
    parser.add_argument("--image_use_source_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_mask_crop_resolution", type=int, default=518)
    parser.add_argument("--projection_grid_transform", default="pixal3d_rotation", choices=["identity", "pixal3d_rotation"])
    parser.add_argument("--condition_adapter_dim", type=int, default=1024)
    parser.add_argument("--condition_adapter_hidden_dim", type=int, default=256)
    parser.add_argument("--condition_adapter_max_views", type=int, default=8)
    parser.add_argument("--condition_adapter_pose_dim", type=int, default=32)
    parser.add_argument("--latent_channels", type=int, default=8)
    parser.add_argument("--condition_projection_token_max_cells", type=int, default=512)
    parser.add_argument("--condition_adapter_use_view_embed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict_input_projection_grid", action="store_true")
    parser.add_argument("--strict_input_projection_channels", type=int, default=6, choices=[3, 6])
    parser.add_argument("--blockwise_global_modulation", action="store_true")
    parser.add_argument("--blockwise_global_target_blocks", type=int, default=4, help="0 means all blocks; positive means final N blocks.")
    parser.add_argument("--blockwise_global_hidden_dim", type=int, default=256)
    parser.add_argument("--blockwise_global_mode", default="bias", choices=["bias", "film"])
    parser.add_argument("--blockwise_token_modulation", action="store_true")
    parser.add_argument("--blockwise_token_target_blocks", type=int, default=4, help="0 means all blocks; positive means final N blocks.")
    parser.add_argument("--blockwise_token_hidden_dim", type=int, default=128)
    parser.add_argument("--blockwise_token_mode", default="film", choices=["bias", "film"])
    parser.add_argument("--flow_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_target_blocks", type=int, default=8, help="0 means all blocks; positive means final N blocks.")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--boundary_refine_steps", type=int, default=0)
    parser.add_argument("--init_mode", default="noise", choices=["noise", "q_splice", "q_splice_noise", "q_splice_t", "stock_latent", "stock_t"])
    parser.add_argument("--start_t", type=float, default=1.0)
    parser.add_argument("--q_splice_noise_scale", type=float, default=0.05)
    parser.add_argument("--include_stock_source", action="store_true")
    parser.add_argument("--stock_steps", type=int, default=0, help="0 means use TRELLIS native sparse sampler steps.")
    parser.add_argument("--stock_cfg_strength", type=float, default=0.0, help="0 means use TRELLIS native sparse sampler cfg.")
    parser.add_argument("--stock_rescale_t", type=float, default=0.0, help="0 means use TRELLIS native sparse sampler rescale_t.")
    parser.add_argument("--rescale_t", type=float, default=1.0)
    parser.add_argument("--cfg_strength", type=float, default=1.0)
    parser.add_argument("--teacher_forced_t", default="0.25,0.5,0.75")
    parser.add_argument("--topk", default="4096,8192,target_unique")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--prior_radius", type=float, default=4.0)
    parser.add_argument("--progress_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
