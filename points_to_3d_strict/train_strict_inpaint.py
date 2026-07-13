#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime
import os
import random
import sys
import time
from pathlib import Path

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
from torch.utils.data import DataLoader

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
    save_json,
    set_flow_blockwise_global_modulation_context,
    set_flow_blockwise_global_modulation_trainable,
    set_flow_blockwise_token_modulation_context,
    set_flow_blockwise_token_modulation_trainable,
    set_flow_lora_trainable,
    strict_collate,
    strict_input_projection_channel_count,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_module_trainable(module: torch.nn.Module, enabled: bool) -> int:
    count = 0
    for p in module.parameters():
        p.requires_grad = bool(enabled)
        count += p.numel()
    return count


def configure_partial_flow_training(flow: torch.nn.Module, args: argparse.Namespace) -> dict:
    for p in flow.parameters():
        p.requires_grad = False
    selected: dict[str, int] = {}
    if bool(args.partial_train_input_layer):
        selected["input_layer"] = set_module_trainable(flow.input_layer, True)
    if bool(args.partial_train_t_embedder):
        selected["t_embedder"] = set_module_trainable(flow.t_embedder, True)
        if getattr(flow, "share_mod", False) and hasattr(flow, "adaLN_modulation"):
            selected["adaLN_modulation"] = set_module_trainable(flow.adaLN_modulation, True)
    if bool(args.partial_train_out_layer):
        selected["out_layer"] = set_module_trainable(flow.out_layer, True)
    n_blocks = max(0, int(args.partial_train_blocks))
    if n_blocks > 0:
        total_blocks = len(flow.blocks)
        for idx in range(max(0, total_blocks - n_blocks), total_blocks):
            selected[f"blocks.{idx}"] = set_module_trainable(flow.blocks[idx], True)
    label = "b0_peripheral_input_t_out"
    if n_blocks > 0:
        label = f"peripheral_plus_last_{n_blocks}_blocks"
    return {
        "enabled": True,
        "label": label,
        "selected": selected,
        "trainable_params": int(sum(p.numel() for p in flow.parameters() if p.requires_grad)),
    }


def build_pipeline_and_flow(args: argparse.Namespace, device: torch.device):
    print(f"[points_to_3d_strict][train] loading TRELLIS weights={args.weights}", flush=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.to(device)
    for name, module in pipeline.models.items():
        module.eval()
        for p in module.parameters():
            p.requires_grad = False
    flow = pipeline.models["sparse_structure_flow_model"].to(device)
    extra_input_channels = strict_input_projection_channel_count(args)
    replace_sparse_flow_input_layer(flow, mask_channels=1 + extra_input_channels)
    flow_lora_inject_summary = {"enabled": False, "modules": {}, "trainable_params": 0}
    if args.train_flow_lora:
        flow_lora_inject_summary = inject_flow_lora(
            flow,
            rank=int(args.lora_rank),
            alpha=float(args.lora_alpha),
            target_blocks=int(args.lora_target_blocks),
        )
    blockwise_summary = {"enabled": False, "selected": {}, "trainable_params": None}
    if args.train_blockwise_global_modulation:
        blockwise_summary = inject_flow_blockwise_global_modulation(
            flow,
            input_dim=blockwise_global_modulation_input_dim(args),
            hidden_dim=int(args.blockwise_global_hidden_dim),
            target_blocks=int(args.blockwise_global_target_blocks),
            mode=str(args.blockwise_global_mode),
        )
    blockwise_token_summary = {"enabled": False, "selected": {}, "trainable_params": None}
    if args.train_blockwise_token_modulation:
        blockwise_token_summary = inject_flow_blockwise_token_modulation(
            flow,
            input_dim=blockwise_token_modulation_input_dim(args, patch_size=int(flow.patch_size)),
            hidden_dim=int(args.blockwise_token_hidden_dim),
            target_blocks=int(args.blockwise_token_target_blocks),
            mode=str(args.blockwise_token_mode),
        )
    adapter = build_condition_adapter(args, cond_dim=int(args.condition_adapter_dim))
    if adapter is not None:
        adapter = adapter.to(device)
    load_strict_checkpoint(flow, args.resume)
    load_condition_adapter_checkpoint(adapter, args.resume)
    flow_partial_summary = {"enabled": False, "selected": {}, "trainable_params": None}
    flow_lora_summary = dict(flow_lora_inject_summary)
    flow.train()
    for p in flow.parameters():
        p.requires_grad = True
    if args.train_condition_adapter_only:
        if adapter is None:
            raise ValueError("--train_condition_adapter_only requires an adapter aggregation such as view_gated or geometry_view_gated")
        for p in flow.parameters():
            p.requires_grad = False
        flow.eval()
    elif args.train_blockwise_global_modulation:
        blockwise_summary = set_flow_blockwise_global_modulation_trainable(flow)
        flow.train()
    elif args.train_blockwise_token_modulation:
        blockwise_token_summary = set_flow_blockwise_token_modulation_trainable(flow)
        flow.train()
    elif args.train_flow_lora:
        flow_lora_summary = set_flow_lora_trainable(
            flow,
            train_input_layer=bool(args.lora_train_input_layer),
            train_t_embedder=bool(args.lora_train_t_embedder),
            train_out_layer=bool(args.lora_train_out_layer),
        )
        flow.train()
    elif args.train_flow_partial:
        flow_partial_summary = configure_partial_flow_training(flow, args)
        flow.train()
    elif args.train_input_layer_only:
        for p in flow.parameters():
            p.requires_grad = False
        for p in flow.input_layer.parameters():
            p.requires_grad = True
    if adapter is not None:
        if args.freeze_condition_adapter:
            for p in adapter.parameters():
                p.requires_grad = False
        adapter.train()
    decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    for p in decoder.parameters():
        p.requires_grad = False
    flow_trainable = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    flow_total = sum(p.numel() for p in flow.parameters())
    adapter_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad) if adapter is not None else 0
    adapter_total = sum(p.numel() for p in adapter.parameters()) if adapter is not None else 0
    trainable_dtypes = sorted(
        {
            str(p.dtype).replace("torch.", "")
            for p in list(flow.parameters()) + ([] if adapter is None else list(adapter.parameters()))
            if p.requires_grad
        }
    )
    print(
        f"[points_to_3d_strict][train] flow trainable={flow_trainable:,} / total={flow_total:,} "
        f"adapter trainable={adapter_trainable:,} / total={adapter_total:,} "
        f"strict_input_projection_channels={extra_input_channels} trainable_dtypes={trainable_dtypes}",
        flush=True,
    )
    if flow_partial_summary["enabled"]:
        print(
            f"[points_to_3d_strict][train] partial_flow label={flow_partial_summary['label']} "
            f"selected={flow_partial_summary['selected']}",
            flush=True,
        )
    if flow_lora_summary.get("enabled"):
        print(
            f"[points_to_3d_strict][train] flow_lora label={flow_lora_summary.get('label')} "
            f"rank={args.lora_rank} alpha={args.lora_alpha} target_blocks={args.lora_target_blocks} "
            f"trainable={flow_lora_summary.get('trainable_params')} "
            f"selected_count={len(flow_lora_summary.get('selected', flow_lora_summary.get('modules', {})))}",
            flush=True,
        )
    if blockwise_summary.get("enabled"):
        print(
            f"[points_to_3d_strict][train] blockwise_global label={blockwise_summary.get('label')} "
            f"mode={blockwise_summary.get('mode')} input_dim={blockwise_summary.get('input_dim')} "
            f"hidden={blockwise_summary.get('hidden_dim')} trainable={blockwise_summary.get('trainable_params')}",
            flush=True,
        )
    if blockwise_token_summary.get("enabled"):
        print(
            f"[points_to_3d_strict][train] blockwise_token label={blockwise_token_summary.get('label')} "
            f"mode={blockwise_token_summary.get('mode')} input_dim={blockwise_token_summary.get('input_dim')} "
            f"hidden={blockwise_token_summary.get('hidden_dim')} trainable={blockwise_token_summary.get('trainable_params')}",
            flush=True,
        )
    args._flow_partial_summary = flow_partial_summary
    args._flow_lora_summary = flow_lora_summary
    args._blockwise_global_summary = blockwise_summary
    args._blockwise_token_summary = blockwise_token_summary
    return pipeline, flow, adapter


def batch_image_cond(
    pipeline,
    batch: dict,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    q_vis: torch.Tensor,
    mask: torch.Tensor,
    adapter: torch.nn.Module | None = None,
) -> torch.Tensor:
    conds = []
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


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline, flow, adapter = build_pipeline_and_flow(args, device)
    t_min = float(args.t_min)
    t_max = float(args.t_max)
    if not (0.0 <= t_min < t_max <= 1.0):
        raise ValueError(f"invalid training t range: t_min={t_min}, t_max={t_max}")
    print(f"[points_to_3d_strict][train] t_range=[{t_min}, {t_max}]", flush=True)
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
    if args.batch_size != 1:
        raise ValueError("strict multiview condition currently requires --batch_size 1")
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=strict_collate,
        drop_last=True,
    )
    trainable_params = [p for p in flow.parameters() if p.requires_grad]
    if adapter is not None:
        trainable_params.extend([p for p in adapter.parameters() if p.requires_grad])
    if not trainable_params:
        raise ValueError("no trainable parameters selected")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=0.0, eps=float(args.adam_eps))
    trainable_dtypes = {p.dtype for p in trainable_params}
    amp_enabled = torch.cuda.is_available() and not args.no_amp
    scaler_enabled = amp_enabled and not args.no_grad_scaler and torch.float16 not in trainable_dtypes
    if amp_enabled and not scaler_enabled and torch.float16 in trainable_dtypes:
        print(
            "[points_to_3d_strict][train] GradScaler disabled because trainable flow parameters are float16; "
            "autocast remains enabled.",
            flush=True,
        )
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    sampler = pipeline.sparse_structure_sampler

    rows: list[dict] = []
    global_step = 0
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_time = datetime.datetime.now().isoformat(timespec="seconds")
    start_wall = time.time()
    while global_step < int(args.max_steps):
        for batch in loader:
            if global_step >= int(args.max_steps):
                break
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
            blockwise_context = None
            if args.train_blockwise_global_modulation:
                blockwise_context = build_blockwise_global_modulation_context(
                    q_vis=q_vis,
                    mask=mask,
                    extra_inputs=extra_inputs,
                    projection_channels=strict_input_projection_channel_count(args),
                )
                set_flow_blockwise_global_modulation_context(flow, blockwise_context)
            if args.train_blockwise_token_modulation:
                blockwise_token_context = build_blockwise_token_modulation_context(
                    q_vis=q_vis,
                    mask=mask,
                    extra_inputs=extra_inputs,
                    projection_channels=strict_input_projection_channel_count(args),
                    patch_size=int(flow.patch_size),
                )
                set_flow_blockwise_token_modulation_context(flow, blockwise_token_context)
            cond = batch_image_cond(
                pipeline,
                batch,
                args,
                device,
                dtype=next(flow.parameters()).dtype,
                q_vis=q_vis,
                mask=mask,
                adapter=adapter,
            )
            if float(args.cfg_drop_prob) > 0 and random.random() < float(args.cfg_drop_prob):
                cond = torch.zeros_like(cond)
            t = random.uniform(t_min, t_max)
            noise = torch.randn_like(q_gt)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                x_inp, gt_v = build_q_comb_t(
                    q_gt=q_gt,
                    q_vis=q_vis,
                    mask=mask,
                    sampler=sampler,
                    t=t,
                    noise=noise,
                    extra_inputs=extra_inputs,
                )
                pred_v = flow(x_inp, torch.tensor([1000.0 * t] * q_gt.shape[0], device=device, dtype=torch.float32), cond)
                sq = (pred_v.float() - gt_v.float()) ** 2
                known = mask.float().expand_as(sq)
                unknown = 1.0 - known
                known_loss = (sq * known).sum() / known.sum().clamp_min(1.0)
                unknown_loss = (sq * unknown).sum() / unknown.sum().clamp_min(1.0)
                loss = float(args.unknown_flow_loss_weight) * unknown_loss + float(args.known_flow_loss_weight) * known_loss
            if not torch.isfinite(loss).all():
                row = {
                    "step": int(global_step + 1),
                    "loss": float(loss.detach().cpu()) if loss.detach().numel() == 1 else "nonfinite",
                    "unknown_flow_loss": float(unknown_loss.detach().cpu()) if torch.isfinite(unknown_loss.detach()).all() else "nonfinite",
                    "known_flow_loss": float(known_loss.detach().cpu()) if torch.isfinite(known_loss.detach()).all() else "nonfinite",
                        "t": float(t),
                        "mask_cell_ratio": float(mask.mean().detach().cpu()),
                        "image_cond_view_count": int(len(batch["image_paths"][0])),
                        "image_cond_aggregation": str(args.image_cond_aggregation),
                        "image_frame_select": str(args.image_frame_select),
                        "projection_token_max_cells": int(args.condition_projection_token_max_cells),
                        "strict_input_projection_grid": int(args.strict_input_projection_grid),
                        "strict_input_projection_channels": int(strict_input_projection_channel_count(args)),
                        "strict_input_channel_count": int(x_inp.shape[1]),
                        "condition_adapter": int(adapter is not None),
                        "train_condition_adapter_only": int(args.train_condition_adapter_only),
                        "train_flow_partial": int(args.train_flow_partial),
                        "train_flow_lora": int(args.train_flow_lora),
                        "train_blockwise_global_modulation": int(args.train_blockwise_global_modulation),
                        "train_blockwise_token_modulation": int(args.train_blockwise_token_modulation),
                        "lora_rank": int(args.lora_rank),
                        "lora_alpha": float(args.lora_alpha),
                        "lora_target_blocks": int(args.lora_target_blocks),
                        "partial_train_blocks": int(args.partial_train_blocks),
                        "cond_token_count": int(cond.shape[1]),
                        "uid": batch["uids"][0],
                        "error": "nonfinite_loss_before_backward",
                }
                rows.append(row)
                write_csv(output_dir / "train_log.csv", rows)
                save_json(output_dir / "train_report.json", {"args": vars(args), "rows": rows, "error": row})
                raise FloatingPointError(f"non-finite loss before backward at step={global_step + 1}: {row}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                if scaler_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            if int(args.finite_check_every) > 0 and (global_step + 1) % int(args.finite_check_every) == 0:
                bad_name = None
                with torch.no_grad():
                    for name, p in flow.named_parameters():
                        if p.requires_grad and not torch.isfinite(p).all():
                            bad_name = f"flow.{name}"
                            break
                    if bad_name is None and adapter is not None:
                        for name, p in adapter.named_parameters():
                            if p.requires_grad and not torch.isfinite(p).all():
                                bad_name = f"condition_adapter.{name}"
                                break
                if bad_name is not None:
                    row = {
                        "step": int(global_step + 1),
                        "loss": float(loss.detach().cpu()),
                        "unknown_flow_loss": float(unknown_loss.detach().cpu()),
                        "known_flow_loss": float(known_loss.detach().cpu()),
                        "t": float(t),
                        "mask_cell_ratio": float(mask.mean().detach().cpu()),
                        "image_cond_view_count": int(len(batch["image_paths"][0])),
                        "image_cond_aggregation": str(args.image_cond_aggregation),
                        "image_frame_select": str(args.image_frame_select),
                        "projection_token_max_cells": int(args.condition_projection_token_max_cells),
                        "strict_input_projection_grid": int(args.strict_input_projection_grid),
                        "strict_input_projection_channels": int(strict_input_projection_channel_count(args)),
                        "strict_input_channel_count": int(x_inp.shape[1]),
                        "condition_adapter": int(adapter is not None),
                        "train_condition_adapter_only": int(args.train_condition_adapter_only),
                        "train_flow_partial": int(args.train_flow_partial),
                        "train_flow_lora": int(args.train_flow_lora),
                        "train_blockwise_global_modulation": int(args.train_blockwise_global_modulation),
                        "train_blockwise_token_modulation": int(args.train_blockwise_token_modulation),
                        "lora_rank": int(args.lora_rank),
                        "lora_alpha": float(args.lora_alpha),
                        "lora_target_blocks": int(args.lora_target_blocks),
                        "partial_train_blocks": int(args.partial_train_blocks),
                        "cond_token_count": int(cond.shape[1]),
                        "uid": batch["uids"][0],
                        "error": f"nonfinite_parameter_after_step:{bad_name}",
                    }
                    rows.append(row)
                    write_csv(output_dir / "train_log.csv", rows)
                    save_json(output_dir / "train_report.json", {"args": vars(args), "rows": rows, "error": row})
                    raise FloatingPointError(f"non-finite trainable parameter after step={global_step + 1}: {bad_name}")
            global_step += 1
            row = {
                "step": int(global_step),
                "loss": float(loss.detach().cpu()),
                "unknown_flow_loss": float(unknown_loss.detach().cpu()),
                "known_flow_loss": float(known_loss.detach().cpu()),
                "unknown_flow_loss_weight": float(args.unknown_flow_loss_weight),
                "known_flow_loss_weight": float(args.known_flow_loss_weight),
                "t": float(t),
                "t_min": t_min,
                "t_max": t_max,
                "mask_cell_ratio": float(mask.mean().detach().cpu()),
                "image_cond_view_count": int(len(batch["image_paths"][0])),
                "image_cond_aggregation": str(args.image_cond_aggregation),
                "image_frame_select": str(args.image_frame_select),
                "projection_token_max_cells": int(args.condition_projection_token_max_cells),
                "strict_input_projection_grid": int(args.strict_input_projection_grid),
                "strict_input_projection_channels": int(strict_input_projection_channel_count(args)),
                "strict_input_channel_count": int(x_inp.shape[1]),
                "condition_adapter": int(adapter is not None),
                "train_condition_adapter_only": int(args.train_condition_adapter_only),
                "train_flow_partial": int(args.train_flow_partial),
                "train_flow_lora": int(args.train_flow_lora),
                "train_blockwise_global_modulation": int(args.train_blockwise_global_modulation),
                "train_blockwise_token_modulation": int(args.train_blockwise_token_modulation),
                "blockwise_global_target_blocks": int(args.blockwise_global_target_blocks),
                "blockwise_global_hidden_dim": int(args.blockwise_global_hidden_dim),
                "blockwise_global_mode": str(args.blockwise_global_mode),
                "blockwise_token_target_blocks": int(args.blockwise_token_target_blocks),
                "blockwise_token_hidden_dim": int(args.blockwise_token_hidden_dim),
                "blockwise_token_mode": str(args.blockwise_token_mode),
                "lora_rank": int(args.lora_rank),
                "lora_alpha": float(args.lora_alpha),
                "lora_target_blocks": int(args.lora_target_blocks),
                "partial_train_blocks": int(args.partial_train_blocks),
                "cond_token_count": int(cond.shape[1]),
                "uid": batch["uids"][0],
            }
            rows.append(row)
            if global_step % max(1, int(args.log_every)) == 0 or global_step == 1:
                elapsed = max(1e-6, time.time() - start_wall)
                steps_per_sec = float(global_step) / elapsed
                remaining = max(0, int(args.max_steps) - int(global_step))
                eta_sec = remaining / max(steps_per_sec, 1e-6)
                print(
                    f"[points_to_3d_strict][train] step={global_step}/{int(args.max_steps)} "
                    f"loss={row['loss']:.6f} "
                    f"unknown={row['unknown_flow_loss']:.6f} known={row['known_flow_loss']:.6f} "
                    f"t={row['t']:.4f} mask={row['mask_cell_ratio']:.4f} tokens={row['cond_token_count']} "
                    f"elapsed={elapsed/60.0:.1f}m eta={eta_sec/60.0:.1f}m",
                    file=sys.stderr,
                    flush=True,
                )
            if global_step % max(1, int(args.save_every)) == 0 or global_step == int(args.max_steps):
                state_dict = {f"flow.{k}": v.detach().cpu() for k, v in flow.state_dict().items()}
                if adapter is not None:
                    state_dict.update({f"condition_adapter.{k}": v.detach().cpu() for k, v in adapter.state_dict().items()})
                ckpt = {
                    "state_dict": state_dict,
                    "args": vars(args),
                    "flow_partial_summary": getattr(args, "_flow_partial_summary", {}),
                    "flow_lora_summary": getattr(args, "_flow_lora_summary", {}),
                    "blockwise_global_summary": getattr(args, "_blockwise_global_summary", {}),
                    "blockwise_token_summary": getattr(args, "_blockwise_token_summary", {}),
                    "step": int(global_step),
                    "started_at": start_time,
                }
                torch.save(ckpt, ckpt_dir / f"strict-points-to-3d-step={global_step}.ckpt")
                torch.save(ckpt, ckpt_dir / "last.ckpt")
                write_csv(output_dir / "train_log.csv", rows)
                save_json(
                    output_dir / "train_report.json",
                    {
                        "args": vars(args),
                        "flow_partial_summary": getattr(args, "_flow_partial_summary", {}),
                        "flow_lora_summary": getattr(args, "_flow_lora_summary", {}),
                        "blockwise_global_summary": getattr(args, "_blockwise_global_summary", {}),
                        "blockwise_token_summary": getattr(args, "_blockwise_token_summary", {}),
                        "rows": rows,
                    },
                )
    print(f"[points_to_3d_strict][train] wrote {ckpt_dir / 'last.ckpt'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict Points-to-3D sparse structure inpainting flow training.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--indices", default="0")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--adam_eps", type=float, default=1e-4)
    parser.add_argument("--finite_check_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_drop_prob", type=float, default=0.0)
    parser.add_argument("--t_min", type=float, default=0.0)
    parser.add_argument("--t_max", type=float, default=1.0)
    parser.add_argument("--unknown_flow_loss_weight", type=float, default=1.0)
    parser.add_argument("--known_flow_loss_weight", type=float, default=0.02)
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
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_input_layer_only", action="store_true", help="Emergency low-memory diagnostic; not the strict paper setting.")
    parser.add_argument("--train_condition_adapter_only", action="store_true", help="Freeze strict flow and train only the multiview condition adapter.")
    parser.add_argument("--train_flow_partial", action="store_true", help="Freeze most of G_inp and train selected sparse-flow modules.")
    parser.add_argument("--partial_train_blocks", type=int, default=0, help="Number of final sparse-flow transformer blocks to unfreeze.")
    parser.add_argument("--partial_train_input_layer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partial_train_t_embedder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partial_train_out_layer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_flow_lora", action="store_true", help="Freeze base G_inp and train LoRA adapters in sparse-flow blocks.")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_target_blocks", type=int, default=8, help="0 means all blocks; positive means final N blocks.")
    parser.add_argument("--lora_train_input_layer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora_train_t_embedder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora_train_out_layer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_blockwise_global_modulation", action="store_true", help="Train zero-init block-wise global geometry modulation adapters.")
    parser.add_argument("--blockwise_global_target_blocks", type=int, default=4, help="0 means all blocks; positive means final N blocks.")
    parser.add_argument("--blockwise_global_hidden_dim", type=int, default=256)
    parser.add_argument("--blockwise_global_mode", default="bias", choices=["bias", "film"])
    parser.add_argument("--train_blockwise_token_modulation", action="store_true", help="Train zero-init block-wise token/spatial geometry modulation adapters.")
    parser.add_argument("--blockwise_token_target_blocks", type=int, default=4, help="0 means all blocks; positive means final N blocks.")
    parser.add_argument("--blockwise_token_hidden_dim", type=int, default=128)
    parser.add_argument("--blockwise_token_mode", default="film", choices=["bias", "film"])
    parser.add_argument("--freeze_condition_adapter", action="store_true", help="Freeze condition adapter while training partial flow.")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_grad_scaler", action="store_true", help="Disable GradScaler. Automatically disabled when trainable params are fp16.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
