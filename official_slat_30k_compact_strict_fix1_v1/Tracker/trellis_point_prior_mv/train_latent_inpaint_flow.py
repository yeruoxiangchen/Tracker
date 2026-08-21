#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, DistributedSampler

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ar_pose_trellis.pipeline import apply_lora_to_ss_flow  # noqa: E402
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline  # noqa: E402

from trellis_point_prior_mv.common import SparsePointPriorCond, parse_indices, resolve_path  # noqa: E402
from trellis_point_prior_mv.eval_latent_splice_sanity import latent_mask_from_prior, normalize_latent_mask  # noqa: E402
from trellis_point_prior_mv.latent_inpaint_image_condition import (  # noqa: E402
    SourceImageResolver,
    encode_image_condition,
    fuse_point_image_condition,
)


def weighted_channel_mean(values: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if values.ndim != 5 or weight.ndim != 5:
        raise ValueError(f"expected [B,C,D,H,W] and [B,1,D,H,W], got {tuple(values.shape)} {tuple(weight.shape)}")
    weight = weight.to(device=values.device, dtype=values.dtype).clamp(0.0, 1.0)
    denom = weight.sum() * values.shape[1] + eps
    return (values * weight).sum() / denom


class LatentInpaintDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest: str | Path,
        *,
        indices: str = "all",
        mask_dilate64: int = 0,
        mask_dilate16: int = 0,
        source_grid_resolution: int = 64,
        latent_grid_resolution: int = 16,
        include_image_paths: bool = False,
        image_max_views: int = 4,
        image_frame_select: str = "uniform",
        image_select_seed: int = 0,
        image_use_source_mask: bool = True,
    ):
        self.manifest_path = Path(manifest)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        samples = payload["samples"]
        selected = parse_indices(indices, len(samples))
        self.samples = [samples[i] for i in selected]
        self.payload = payload
        self.latent_root = payload.get("latent_root") or str(self.manifest_path.parent)
        self.mask_dilate64 = int(mask_dilate64)
        self.mask_dilate16 = int(mask_dilate16)
        self.source_grid_resolution = int(source_grid_resolution or payload.get("source_grid_resolution", 64))
        self.latent_grid_resolution = int(latent_grid_resolution or payload.get("latent_grid_resolution", 16))
        self.include_image_paths = bool(include_image_paths)
        self.image_max_views = int(image_max_views)
        self.image_frame_select = str(image_frame_select)
        self.image_select_seed = int(image_select_seed)
        self.image_use_source_mask = bool(image_use_source_mask)
        self.image_resolver = SourceImageResolver(payload.get("source_manifest")) if self.include_image_paths else None
        print(
            f"[LatentInpaintDataset] {len(self.samples)} samples from {manifest} "
            f"d64={self.mask_dilate64} d16={self.mask_dilate16} image_cond={self.include_image_paths}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        latent_path = resolve_path(self.latent_root, sample["latent_npz"])
        with np.load(latent_path) as data:
            q_gt = np.asarray(data["q_gt"], dtype=np.float32)
            q_vis = np.asarray(data["q_vis"], dtype=np.float32)
            saved_m_s = normalize_latent_mask(
                np.asarray(data["m_s"], dtype=np.float32),
                latent_resolution=self.latent_grid_resolution,
            )
            prior_coords = np.asarray(data["prior_coords"], dtype=np.int32)
            target_coords = np.asarray(data["target_coords"], dtype=np.int32)
        if self.mask_dilate64 == 0 and self.mask_dilate16 == 0:
            mask = saved_m_s
        else:
            mask = latent_mask_from_prior(
                prior_coords,
                mask_dilate64=self.mask_dilate64,
                mask_dilate16=self.mask_dilate16,
                source_resolution=self.source_grid_resolution,
                latent_resolution=self.latent_grid_resolution,
            )
        item = {
            "uid": str(sample.get("uid", index)),
            "q_gt": torch.from_numpy(q_gt.astype(np.float32)),
            "q_vis": torch.from_numpy(q_vis.astype(np.float32)),
            "m_s": torch.from_numpy(mask.astype(np.float32)),
            "saved_m_s": torch.from_numpy(saved_m_s.astype(np.float32)),
            "prior_coords": torch.from_numpy(prior_coords.astype(np.int64)),
            "target_coords": torch.from_numpy(target_coords.astype(np.int64)),
            "latent_path": str(latent_path),
        }
        if self.include_image_paths:
            if self.image_resolver is None:
                raise RuntimeError("image resolver was not initialized")
            image_paths, mask_paths = self.image_resolver.image_mask_paths(
                sample,
                max_views=self.image_max_views,
                frame_select=self.image_frame_select,
                seed=self.image_select_seed,
            )
            item["image_paths"] = image_paths
            item["image_mask_paths"] = mask_paths if self.image_use_source_mask else [None for _ in image_paths]
        return item


def latent_inpaint_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "q_gt": torch.stack([item["q_gt"] for item in batch], dim=0),
        "q_vis": torch.stack([item["q_vis"] for item in batch], dim=0),
        "m_s": torch.stack([item["m_s"] for item in batch], dim=0),
        "saved_m_s": torch.stack([item["saved_m_s"] for item in batch], dim=0),
        "uids": [item["uid"] for item in batch],
        "latent_paths": [item["latent_path"] for item in batch],
        "image_paths": [item.get("image_paths", []) for item in batch],
        "image_mask_paths": [item.get("image_mask_paths", []) for item in batch],
    }


class LatentInpaintFlowTrainer(pl.LightningModule):
    def __init__(
        self,
        ss_flow_model,
        ss_cond: SparsePointPriorCond,
        ss_sampler,
        pipeline,
        *,
        lr: float,
        cfg_drop_prob: float,
        unknown_flow_loss_weight: float,
        known_flow_loss_weight: float,
        unknown_x0_loss_weight: float,
        known_x0_loss_weight: float,
        cond_use_masked_q_vis: bool,
        use_image_cond: bool,
        image_cond_aggregation: str,
        image_preprocess: bool,
        image_use_source_mask: bool,
        image_mask_crop_resolution: int,
        cond_fusion: str,
    ):
        super().__init__()
        self.ss_flow_model = ss_flow_model
        self.ss_cond = ss_cond
        self.ss_sampler = ss_sampler
        self.pipeline = pipeline
        self.lr = float(lr)
        self.cfg_drop_prob = float(cfg_drop_prob)
        self.unknown_flow_loss_weight = float(unknown_flow_loss_weight)
        self.known_flow_loss_weight = float(known_flow_loss_weight)
        self.unknown_x0_loss_weight = float(unknown_x0_loss_weight)
        self.known_x0_loss_weight = float(known_x0_loss_weight)
        self.cond_use_masked_q_vis = bool(cond_use_masked_q_vis)
        self.use_image_cond = bool(use_image_cond)
        self.image_cond_aggregation = str(image_cond_aggregation)
        self.image_preprocess = bool(image_preprocess)
        self.image_use_source_mask = bool(image_use_source_mask)
        self.image_mask_crop_resolution = int(image_mask_crop_resolution)
        self.cond_fusion = str(cond_fusion)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(device, non_blocking=False) if torch.is_tensor(value) else value
        return moved

    def _encode_batch_image_condition(
        self,
        image_paths_batch: list[list[str]],
        image_mask_paths_batch: list[list[str | None]] | None,
        *,
        device,
        dtype,
    ) -> torch.Tensor | None:
        if not self.use_image_cond:
            return None
        conds = []
        image_mask_paths_batch = image_mask_paths_batch or [[] for _ in image_paths_batch]
        for image_paths, mask_paths in zip(image_paths_batch, image_mask_paths_batch):
            conds.append(
                encode_image_condition(
                    self.pipeline,
                    list(image_paths),
                    device=device,
                    dtype=dtype,
                    aggregation=self.image_cond_aggregation,
                    preprocess=self.image_preprocess,
                    mask_paths=list(mask_paths),
                    use_source_mask=self.image_use_source_mask,
                    mask_crop_resolution=self.image_mask_crop_resolution,
                )
            )
        token_counts = {int(cond.shape[1]) for cond in conds}
        if len(token_counts) != 1:
            raise ValueError(f"batched image condition has variable token counts: {sorted(token_counts)}")
        return torch.cat(conds, dim=0)

    def build_condition(
        self,
        q_vis: torch.Tensor,
        mask: torch.Tensor,
        image_paths_batch: list[list[str]] | None,
        image_mask_paths_batch: list[list[str | None]] | None,
    ) -> torch.Tensor:
        confidence = mask.clamp(0.0, 1.0)
        cond_latent = q_vis * mask if self.cond_use_masked_q_vis else q_vis
        point_cond = self.ss_cond(cond_latent, mask, confidence)
        image_cond = self._encode_batch_image_condition(
            image_paths_batch or [],
            image_mask_paths_batch,
            device=point_cond.device,
            dtype=point_cond.dtype,
        )
        cond = fuse_point_image_condition(point_cond, image_cond, self.cond_fusion)
        if random.random() < self.cfg_drop_prob:
            cond = torch.zeros_like(cond)
        return cond

    def training_step(self, batch, batch_idx):
        q_gt = batch["q_gt"].to(self.device, dtype=torch.float32)
        q_vis = batch["q_vis"].to(self.device, dtype=torch.float32)
        mask = batch["m_s"].to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
        unknown = 1.0 - mask
        target = q_gt * unknown + q_vis * mask
        cond = self.build_condition(q_vis, mask, batch.get("image_paths"), batch.get("image_mask_paths"))

        t = torch.rand(1).item()
        noise = torch.randn_like(target)
        x_t, gt_v = self.ss_sampler._get_model_gt(target, t, noise)
        t_tensor = torch.tensor([1000.0 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        pred_v = self.ss_flow_model(x_t, t_tensor, cond)
        flow_mse = torch.nan_to_num(F.mse_loss(pred_v, gt_v, reduction="none"), nan=0.0, posinf=1e4, neginf=1e4)
        loss_unknown_flow = weighted_channel_mean(flow_mse, unknown)
        loss_known_flow = weighted_channel_mean(flow_mse, mask)

        pred_x0 = self.ss_sampler._pred_to_xstart(x_t, t, pred_v)
        x0_mse = torch.nan_to_num(F.mse_loss(pred_x0, target, reduction="none"), nan=0.0, posinf=1e4, neginf=1e4)
        loss_unknown_x0 = weighted_channel_mean(x0_mse, unknown)
        loss_known_x0 = weighted_channel_mean(x0_mse, mask)

        with torch.no_grad():
            pred_vs_gt_unknown_l1 = weighted_channel_mean(torch.abs(pred_x0 - q_gt), unknown)
            pred_vs_vis_known_l1 = weighted_channel_mean(torch.abs(pred_x0 - q_vis), mask)
            q_vis_vs_gt_known_l1 = weighted_channel_mean(torch.abs(q_vis - q_gt), mask)
            q_vis_vs_gt_unknown_l1 = weighted_channel_mean(torch.abs(q_vis - q_gt), unknown)

        loss_total = (
            self.unknown_flow_loss_weight * loss_unknown_flow
            + self.known_flow_loss_weight * loss_known_flow
            + self.unknown_x0_loss_weight * loss_unknown_x0
            + self.known_x0_loss_weight * loss_known_x0
        )

        self.log("train_loss", loss_total, prog_bar=True, sync_dist=True)
        self.log("loss_unknown_flow", loss_unknown_flow, prog_bar=False, sync_dist=True)
        self.log("loss_known_flow", loss_known_flow, prog_bar=False, sync_dist=True)
        self.log("loss_unknown_x0", loss_unknown_x0, prog_bar=False, sync_dist=True)
        self.log("loss_known_x0", loss_known_x0, prog_bar=False, sync_dist=True)
        self.log("mask_ratio", mask.mean(), prog_bar=False, sync_dist=True)
        self.log("cond_token_count", torch.tensor(float(cond.shape[1]), device=self.device), prog_bar=False, sync_dist=True)
        self.log("pred_vs_gt_unknown_l1", pred_vs_gt_unknown_l1, prog_bar=False, sync_dist=True)
        self.log("pred_vs_vis_known_l1", pred_vs_vis_known_l1, prog_bar=False, sync_dist=True)
        self.log("q_vis_vs_gt_known_l1", q_vis_vs_gt_known_l1, prog_bar=False, sync_dist=True)
        self.log("q_vis_vs_gt_unknown_l1", q_vis_vs_gt_unknown_l1, prog_bar=False, sync_dist=True)
        return loss_total

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        checkpoint["state_dict"] = {
            k: v for k, v in state_dict.items() if k.startswith("ss_flow_model.") or k.startswith("ss_cond.")
        }

    def configure_optimizers(self):
        params = list(self.ss_cond.parameters()) + [p for p in self.ss_flow_model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=0.0)


def build_model(args: argparse.Namespace, local_rank: int) -> LatentInpaintFlowTrainer:
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.to(device)
    for module in pipeline.models.values():
        if hasattr(module, "eval"):
            module.eval()
        if hasattr(module, "parameters"):
            for p in module.parameters():
                p.requires_grad = False

    ss_flow_model = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    for p in ss_flow_model.parameters():
        p.requires_grad = False
    ss_flow_model = apply_lora_to_ss_flow(ss_flow_model, r=args.lora_rank, alpha=args.lora_alpha)

    ss_cond = SparsePointPriorCond(
        latent_channels=args.latent_channels,
        cond_channels=args.cond_channels,
        grid_resolution=args.latent_grid_resolution,
    ).to(device).train()

    model = LatentInpaintFlowTrainer(
        ss_flow_model=ss_flow_model,
        ss_cond=ss_cond,
        ss_sampler=pipeline.sparse_structure_sampler,
        pipeline=pipeline,
        lr=args.lr,
        cfg_drop_prob=args.cfg_drop_prob,
        unknown_flow_loss_weight=args.unknown_flow_loss_weight,
        known_flow_loss_weight=args.known_flow_loss_weight,
        unknown_x0_loss_weight=args.unknown_x0_loss_weight,
        known_x0_loss_weight=args.known_x0_loss_weight,
        cond_use_masked_q_vis=not args.cond_use_full_q_vis,
        use_image_cond=args.use_image_cond,
        image_cond_aggregation=args.image_cond_aggregation,
        image_preprocess=args.image_preprocess,
        image_use_source_mask=args.image_use_source_mask,
        image_mask_crop_resolution=args.image_mask_crop_resolution,
        cond_fusion=args.cond_fusion,
    )
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu")
        missing, unexpected = model.load_state_dict(state.get("state_dict", state), strict=False)
        print(
            f"[train_latent_inpaint_flow] loaded {args.resume}: "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train masked latent inpainting flow from q_vis/m_s to q_gt unknown region.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume_full_state", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--accum_batches", type=int, default=1)
    parser.add_argument("--ckpt_every_n_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_drop_prob", type=float, default=0.05)
    parser.add_argument("--mask_dilate64", type=int, default=0)
    parser.add_argument("--mask_dilate16", type=int, default=0)
    parser.add_argument("--source_grid_resolution", type=int, default=64)
    parser.add_argument("--latent_grid_resolution", type=int, default=16)
    parser.add_argument("--unknown_flow_loss_weight", type=float, default=1.0)
    parser.add_argument("--known_flow_loss_weight", type=float, default=0.25)
    parser.add_argument("--unknown_x0_loss_weight", type=float, default=0.25)
    parser.add_argument("--known_x0_loss_weight", type=float, default=0.10)
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
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--latent_channels", type=int, default=8)
    parser.add_argument("--cond_channels", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
    pl.seed_everything(int(args.seed) + local_rank, workers=True)
    if args.use_image_cond and args.batch_size != 1:
        raise ValueError("image condition currently requires --batch_size 1 to avoid variable multi-view token lengths")

    model = build_model(args, local_rank)
    dataset = LatentInpaintDataset(
        args.manifest,
        indices=args.indices,
        mask_dilate64=args.mask_dilate64,
        mask_dilate16=args.mask_dilate16,
        source_grid_resolution=args.source_grid_resolution,
        latent_grid_resolution=args.latent_grid_resolution,
        include_image_paths=args.use_image_cond,
        image_max_views=args.image_max_views,
        image_frame_select=args.image_frame_select,
        image_select_seed=args.image_select_seed,
        image_use_source_mask=args.image_use_source_mask,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=world_rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=latent_inpaint_collate,
        drop_last=True,
    )
    os.makedirs(args.save_dir, exist_ok=True)
    logger = TensorBoardLogger(args.save_dir, name="tb", version=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    ckpt_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="ss-latent-inpaint-{epoch:02d}-{step}",
        save_top_k=-1,
        every_n_train_steps=args.ckpt_every_n_steps,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=world_size if torch.cuda.is_available() else 1,
        strategy="ddp_find_unused_parameters_true" if world_size > 1 else "auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=logger,
        callbacks=[ckpt_cb],
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        limit_train_batches=args.limit_train_batches,
        accumulate_grad_batches=args.accum_batches,
        log_every_n_steps=10,
    )
    trainer.fit(model, loader, ckpt_path=args.resume if args.resume_full_state else None)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
