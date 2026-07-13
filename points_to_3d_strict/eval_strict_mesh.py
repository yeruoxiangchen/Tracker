#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
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

from points_to_3d_strict.eval_strict_inpaint import decode_logits, encode_cond_for_batch  # noqa: E402
from points_to_3d_strict.strict_core import (  # noqa: E402
    StrictLatentDataset,
    blockwise_global_modulation_input_dim,
    blockwise_token_modulation_input_dim,
    build_blockwise_global_modulation_context,
    build_blockwise_token_modulation_context,
    build_condition_adapter,
    build_strict_input_projection_features,
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
from trellis_point_prior_mv.eval_mesh_frozen_downstream import (  # noqa: E402
    apply_mask_and_crop,
    coords_np_to_torch,
    mesh_artifact_metrics_from_obj,
    mesh_basic_metrics,
    mesh_target_distance_metrics,
    prepare_cond,
    sample_slat_mesh,
)
from trellis_point_prior_mv.eval_sparse_inpaint import topk_coords_from_logits  # noqa: E402
from trellis_point_prior_mv.eval_sparse_vae_sanity import coords4, threshold_coords_from_logits  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_modes(text: str) -> list[str]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(part)
    if not out:
        raise ValueError("empty modes")
    return out


def load_models(args: argparse.Namespace, device: torch.device):
    print(f"[points_to_3d_strict][mesh] loading TRELLIS weights={args.weights}", flush=True)
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


def coords_from_latent(decoder, latent: torch.Tensor, *, decode_mode: str, target_unique: int, threshold: float) -> np.ndarray:
    logits = decode_logits(decoder, latent)
    mode = str(decode_mode).lower()
    if mode == "threshold":
        return threshold_coords_from_logits(logits, float(threshold))
    if mode == "target_unique":
        return topk_coords_from_logits(logits, int(target_unique))
    if mode.startswith("topk_"):
        return topk_coords_from_logits(logits, int(mode.split("_", 1)[1]))
    raise ValueError(f"unsupported coord_decode={decode_mode!r}")


def needs_stock_latent(args: argparse.Namespace) -> bool:
    modes = {m.strip() for m in str(args.modes).split(",") if m.strip()}
    return str(args.init_mode) in {"stock_latent", "stock_t"} or bool({"q_stock", "q_stock_clamp"} & modes)


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
        print(f"[points_to_3d_strict][mesh] stock latent sample={order} shape={tuple(latent.shape)}", flush=True)
    return stock


def effective_start_t(args: argparse.Namespace) -> float:
    start_t = float(args.start_t)
    rescale_t = float(args.rescale_t)
    return float(rescale_t * start_t / (1.0 + (rescale_t - 1.0) * start_t))


def build_initial_sample(args: argparse.Namespace, sampler, q_vis: torch.Tensor, q_splice: torch.Tensor, q_stock: torch.Tensor | None) -> torch.Tensor:
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


def sample_strict_latent(
    flow,
    sampler,
    q_vis,
    mask,
    cond,
    args,
    seed: int,
    initial_sample: torch.Tensor,
    extra_inputs: torch.Tensor | None = None,
) -> torch.Tensor:
    torch.manual_seed(int(seed))
    return sample_points_to_3d_strict(
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
    rows: list[dict[str, Any]] = []
    modes = parse_modes(args.modes)
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
            f"[points_to_3d_strict][mesh] blockwise_global label={blockwise_summary.get('label')} "
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
            f"[points_to_3d_strict][mesh] blockwise_token label={blockwise_token_summary.get('label')} "
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
            f"[points_to_3d_strict][mesh] flow_lora label={summary.get('label')} "
            f"rank={args.lora_rank} alpha={args.lora_alpha} target_blocks={args.lora_target_blocks}",
            flush=True,
        )
    load_strict_checkpoint(flow, args.checkpoint)
    load_condition_adapter_checkpoint(adapter, args.checkpoint)
    flow.eval()
    if adapter is not None:
        adapter.eval()
    for order, batch in enumerate(loader):
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
        cond_tensor = encode_cond_for_batch(pipeline, batch, args, device, dtype=next(flow.parameters()).dtype, adapter=adapter)
        target_coords = batch["target_coords"][0]
        target_unique = int(coords4(target_coords).shape[0])
        q_splice = mask * q_vis + (1.0 - mask) * q_gt
        q_stock = stock_latents.get(order)
        q_stock_t = q_stock.to(device=device, dtype=torch.float32) if q_stock is not None else None
        q_stock_clamp = mask * q_vis + (1.0 - mask) * q_stock_t if q_stock_t is not None else None
        initial_sample = build_initial_sample(args, pipeline.sparse_structure_sampler, q_vis, q_splice, q_stock_t)
        q_pred = sample_strict_latent(
            flow,
            pipeline.sparse_structure_sampler,
            q_vis,
            mask,
            cond_tensor,
            args,
            int(args.seed + order * 1009),
            initial_sample,
            extra_inputs=extra_inputs,
        )

        images = [
            apply_mask_and_crop(Path(img), Path(mask_path), int(args.resolution))
            for img, mask_path in zip(batch["image_paths"][0], batch["image_mask_paths"][0])
        ]
        cond_dict, cond_count = prepare_cond(pipeline, images, args.cond_mode)
        sample_dir = output_dir / f"{order:04d}_{uid[:12]}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"[points_to_3d_strict][mesh] sample={order} uid={uid} modes={modes}", flush=True)

        latent_sources = {
            "q_pred": q_pred,
            "q_splice": q_splice,
            "q_vis": q_vis,
            "q_gt": q_gt,
        }
        if q_stock_t is not None:
            latent_sources["q_stock"] = q_stock_t
        if q_stock_clamp is not None:
            latent_sources["q_stock_clamp"] = q_stock_clamp
        for mode in modes:
            if mode not in latent_sources:
                raise ValueError(f"unsupported mode={mode}; use q_pred,q_splice,q_vis,q_gt,q_stock,q_stock_clamp")
            latent = latent_sources[mode]
            coords = coords_from_latent(
                decoder,
                latent,
                decode_mode=args.coord_decode,
                target_unique=target_unique,
                threshold=float(args.threshold),
            )
            coords_t = coords_np_to_torch(coords, device=device, max_coords=int(args.max_coords), seed=int(args.seed + order))
            mode_dir = sample_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mode_dir / "coords.npz", coords=coords.astype(np.int32))
            row: dict[str, Any] = {
                "sample_order": int(order),
                "uid": uid,
                "mode": mode,
                "coord_decode": args.coord_decode,
                "coord_count": int(coords4(coords).shape[0]),
                "target_unique": int(target_unique),
                "image_cond_view_count": int(len(images)),
                "strict_image_cond_view_count": int(len(batch["image_paths"][0])),
                "strict_image_cond_aggregation": str(args.image_cond_aggregation),
                "strict_image_frame_select": str(args.image_frame_select),
                "strict_image_cond_token_count": int(cond_tensor.shape[1]),
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
                "cond_mode": args.cond_mode,
                "init_mode": str(args.init_mode),
                "start_t": float(args.start_t),
                "effective_start_t": float(effective_start_t(args)),
            }
            try:
                mesh = sample_slat_mesh(pipeline, cond_dict, cond_count, coords_t, args, int(args.seed + order * 3571))
                obj_path = mode_dir / "mesh.obj"
                tri = mesh.to_trimesh(transform_pose=False)
                tri.export(obj_path)
                row.update(mesh_basic_metrics(mesh))
                row.update(mesh_artifact_metrics_from_obj(obj_path))
                row.update(mesh_target_distance_metrics(mesh, target_coords, int(args.mesh_eval_samples), int(args.seed + order)))
                row["mesh_obj"] = str(obj_path)
            except Exception as exc:
                row.update({"mesh_success": 0, "mesh_error": repr(exc)})
            rows.append(row)
    report = {"args": vars(args), "rows": rows}
    save_json(output_dir / "report.json", report)
    write_csv(output_dir / "report.csv", rows)
    print(f"[points_to_3d_strict][mesh] wrote {output_dir / 'report.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict Points-to-3D q_pred -> TRELLIS slat/mesh eval.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--indices", default="0")
    parser.add_argument("--modes", default="q_pred,q_splice")
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
    parser.add_argument("--stock_steps", type=int, default=0, help="0 means use TRELLIS native sparse sampler steps.")
    parser.add_argument("--stock_cfg_strength", type=float, default=0.0, help="0 means use TRELLIS native sparse sampler cfg.")
    parser.add_argument("--stock_rescale_t", type=float, default=0.0, help="0 means use TRELLIS native sparse sampler rescale_t.")
    parser.add_argument("--rescale_t", type=float, default=1.0)
    parser.add_argument("--cfg_strength", type=float, default=1.0)
    parser.add_argument("--coord_decode", default="threshold", help="threshold, target_unique, or topk_N")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max_coords", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--cond_mode", default="multi_stochastic", choices=["first", "mean", "multi_stochastic"])
    parser.add_argument("--slat_steps", type=int, default=12)
    parser.add_argument("--slat_guidance_strength", type=float, default=7.5)
    parser.add_argument("--slat_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--slat_rescale_t", type=float, default=3.0)
    parser.add_argument("--mesh_eval_samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
