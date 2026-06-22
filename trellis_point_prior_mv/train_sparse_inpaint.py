#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

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
from trellis import models as trellis_models  # noqa: E402
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline  # noqa: E402

from trellis_point_prior_mv.common import (  # noqa: E402
    SparsePointPriorCond,
    coords_to_batched_occ,
    load_target_latent,
    partial_latent_stats,
    resolve_path,
)


class PointPriorManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: str):
        import json

        self.manifest_path = Path(manifest)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.samples = payload["samples"]
        self.prior_root = payload.get("prior_root") or str(self.manifest_path.parent)
        print(f"[PointPriorManifestDataset] {len(self.samples)} samples from {manifest}", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        latent_path = Path(sample["ss_latent"])
        if not latent_path.is_absolute():
            latent_path = resolve_path(None, latent_path)
        z, target_coords = load_target_latent(latent_path)
        prior_path = resolve_path(self.prior_root, sample["prior_npz"])
        with np.load(prior_path) as data:
            prior_coords = np.asarray(data["prior_coords"], dtype=np.int64)
            prior_conf = np.asarray(data["prior_conf"], dtype=np.float32) if "prior_conf" in data else np.ones((prior_coords.shape[0],), dtype=np.float32)
        return {
            "uid": str(sample.get("uid", index)),
            "x_0": torch.from_numpy(z.astype(np.float32)),
            "target_coords": torch.from_numpy(target_coords.astype(np.int64)),
            "prior_coords": torch.from_numpy(prior_coords.astype(np.int64)),
            "prior_conf": torch.from_numpy(prior_conf.astype(np.float32)),
            "latent_path": str(latent_path),
            "prior_path": str(prior_path),
        }


def point_prior_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    prior_coords = []
    prior_conf = []
    target_coords = []
    for batch_idx, item in enumerate(batch):
        pc = item["prior_coords"].long()
        if pc.numel():
            prior_coords.append(torch.cat([torch.full((pc.shape[0], 1), batch_idx, dtype=torch.long), pc[:, -3:]], dim=1))
            prior_conf.append(item["prior_conf"].float().reshape(-1))
        tc = item["target_coords"].long()
        if tc.numel():
            target_coords.append(torch.cat([torch.full((tc.shape[0], 1), batch_idx, dtype=torch.long), tc[:, -3:]], dim=1))
    return {
        "x_0": torch.stack([item["x_0"] for item in batch], dim=0),
        "prior_coords": torch.cat(prior_coords, dim=0) if prior_coords else torch.zeros((0, 4), dtype=torch.long),
        "prior_conf": torch.cat(prior_conf, dim=0) if prior_conf else torch.zeros((0,), dtype=torch.float32),
        "target_coords": torch.cat(target_coords, dim=0) if target_coords else torch.zeros((0, 4), dtype=torch.long),
        "uids": [item["uid"] for item in batch],
        "latent_paths": [item["latent_path"] for item in batch],
        "prior_paths": [item["prior_path"] for item in batch],
    }


class SparseInpaintTrainer(pl.LightningModule):
    def __init__(
        self,
        ss_flow_model,
        ss_cond: SparsePointPriorCond,
        ss_encoder,
        ss_sampler,
        *,
        lr: float,
        cfg_drop_prob: float,
    ):
        super().__init__()
        self.ss_flow_model = ss_flow_model
        self.ss_cond = ss_cond
        self.ss_encoder = ss_encoder
        self.ss_sampler = ss_sampler
        self.lr = float(lr)
        self.cfg_drop_prob = float(cfg_drop_prob)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device, non_blocking=False)
            else:
                moved[key] = value
        return moved

    def _encoder_dtype(self) -> torch.dtype:
        return next(self.ss_encoder.parameters()).dtype

    def get_input(self, batch):
        targets = batch["x_0"].to(self.device, dtype=torch.float32)
        b = int(targets.shape[0])
        prior_coords = batch["prior_coords"].to(self.device, dtype=torch.long)
        prior_conf = batch["prior_conf"].to(self.device, dtype=torch.float32)
        with torch.no_grad():
            partial_occ = coords_to_batched_occ(
                prior_coords,
                b,
                resolution=64,
                device=self.device,
                dtype=self._encoder_dtype(),
            )
            partial_latent = self.ss_encoder(partial_occ, sample_posterior=False).to(torch.float32)
        latent_mask, confidence = partial_latent_stats(
            prior_coords,
            b,
            weights=prior_conf,
            latent_resolution=partial_latent.shape[-1],
            source_resolution=64,
            device=self.device,
        )
        partial_latent = partial_latent * latent_mask * confidence
        cond = self.ss_cond(partial_latent, latent_mask, confidence)
        if random.random() < self.cfg_drop_prob:
            cond = torch.zeros_like(cond)
        noise = torch.randn_like(targets)
        stats = {
            "prior_points": float(prior_coords.shape[0] / max(b, 1)),
            "latent_mask_ratio": float(latent_mask.mean().detach().cpu()),
            "latent_conf_mean": float(confidence.mean().detach().cpu()),
        }
        return targets, cond, noise, stats

    def training_step(self, batch, batch_idx):
        t = torch.rand(1).item()
        targets, cond, noise, stats = self.get_input(batch)
        x_t, gt_v = self.ss_sampler._get_model_gt(targets, t, noise)
        t_tensor = torch.tensor([1000.0 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        pred_v = self.ss_flow_model(x_t, t_tensor, cond)
        loss = F.mse_loss(pred_v, gt_v, reduction="none")
        loss_flow = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4).mean()
        self.log("train_loss", loss_flow, prog_bar=True, sync_dist=True)
        self.log("prior_points", stats["prior_points"], prog_bar=False, sync_dist=True)
        self.log("latent_mask_ratio", stats["latent_mask_ratio"], prog_bar=False, sync_dist=True)
        self.log("latent_conf_mean", stats["latent_conf_mean"], prog_bar=False, sync_dist=True)
        return loss_flow

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


def build_model(args: argparse.Namespace, local_rank: int) -> SparseInpaintTrainer:
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.to(device)

    ss_flow_model = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    for p in ss_flow_model.parameters():
        p.requires_grad = False
    ss_flow_model = apply_lora_to_ss_flow(ss_flow_model, r=args.lora_rank, alpha=args.lora_alpha)

    ss_encoder = trellis_models.from_pretrained(
        f"{args.weights}/ckpts/ss_enc_conv3d_16l8_fp16"
        if os.path.isdir(args.weights)
        else f"{args.weights}/ckpts/ss_enc_conv3d_16l8_fp16"
    ).to(device).eval()
    for p in ss_encoder.parameters():
        p.requires_grad = False

    ss_cond = SparsePointPriorCond(
        latent_channels=args.latent_channels,
        cond_channels=args.cond_channels,
        grid_resolution=args.latent_grid_resolution,
    ).to(device).train()
    model = SparseInpaintTrainer(
        ss_flow_model=ss_flow_model,
        ss_cond=ss_cond,
        ss_encoder=ss_encoder,
        ss_sampler=pipeline.sparse_structure_sampler,
        lr=args.lr,
        cfg_drop_prob=args.cfg_drop_prob,
    )
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu")
        missing, unexpected = model.load_state_dict(state.get("state_dict", state), strict=False)
        print(
            f"[train_sparse_inpaint] loaded {args.resume}: "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--save_dir", required=True)
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
    parser.add_argument("--cfg_drop_prob", type=float, default=0.1)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--latent_channels", type=int, default=8)
    parser.add_argument("--latent_grid_resolution", type=int, default=16)
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
    pl.seed_everything(42 + local_rank, workers=True)

    model = build_model(args, local_rank)
    dataset = PointPriorManifestDataset(args.manifest)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=world_rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=point_prior_collate,
        drop_last=True,
    )
    os.makedirs(args.save_dir, exist_ok=True)
    logger = TensorBoardLogger(args.save_dir, name="tb", version=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    ckpt_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="ss-pointprior-{epoch:02d}-{step}",
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
