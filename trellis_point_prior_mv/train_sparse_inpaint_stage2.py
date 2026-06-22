#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

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
    partial_latent_stats,
)
from trellis_point_prior_mv.train_sparse_inpaint import (  # noqa: E402
    PointPriorManifestDataset,
    point_prior_collate,
)


def weighted_channel_mean(values: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if values.ndim != 5 or weight.ndim != 5:
        raise ValueError(f"expected [B,C,D,H,W] and [B,1,D,H,W], got {tuple(values.shape)} {tuple(weight.shape)}")
    weight = weight.to(device=values.device, dtype=values.dtype).clamp(0.0, 1.0)
    denom = weight.sum() * values.shape[1] + eps
    return (values * weight).sum() / denom


class SparseInpaintStage2Trainer(pl.LightningModule):
    def __init__(
        self,
        ss_flow_model,
        ss_cond: SparsePointPriorCond,
        ss_encoder,
        ss_decoder,
        ss_sampler,
        *,
        lr: float,
        cfg_drop_prob: float,
        known_flow_loss_weight: float,
        known_x0_loss_weight: float,
        known_conf_power: float,
        known_use_confidence: bool,
        anti_overfill_loss_weight: float,
        anti_overfill_margin: float,
        ranking_loss_weight: float,
        ranking_margin: float,
        ranking_negative_modes: str,
        ranking_outside_weight: float,
        ranking_observed_weight: float,
        ranking_wrong_support_weight: float,
        ranking_target_support_weight: float,
    ):
        super().__init__()
        self.ss_flow_model = ss_flow_model
        self.ss_cond = ss_cond
        self.ss_encoder = ss_encoder
        self.ss_decoder = ss_decoder
        self.ss_sampler = ss_sampler
        self.lr = float(lr)
        self.cfg_drop_prob = float(cfg_drop_prob)
        self.known_flow_loss_weight = float(known_flow_loss_weight)
        self.known_x0_loss_weight = float(known_x0_loss_weight)
        self.known_conf_power = float(known_conf_power)
        self.known_use_confidence = bool(known_use_confidence)
        self.anti_overfill_loss_weight = float(anti_overfill_loss_weight)
        self.anti_overfill_margin = float(anti_overfill_margin)
        self.ranking_loss_weight = float(ranking_loss_weight)
        self.ranking_margin = float(ranking_margin)
        self.ranking_negative_modes = tuple(m.strip() for m in str(ranking_negative_modes).split(",") if m.strip())
        self.ranking_outside_weight = float(ranking_outside_weight)
        self.ranking_observed_weight = float(ranking_observed_weight)
        self.ranking_wrong_support_weight = float(ranking_wrong_support_weight)
        self.ranking_target_support_weight = float(ranking_target_support_weight)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(device, non_blocking=False) if torch.is_tensor(value) else value
        return moved

    def _encoder_dtype(self) -> torch.dtype:
        return next(self.ss_encoder.parameters()).dtype

    def _decoder_dtype(self) -> torch.dtype:
        return next(self.ss_decoder.parameters()).dtype

    def decode_occ_logits(self, latent: torch.Tensor) -> torch.Tensor:
        self.ss_decoder.eval()
        logits = self.ss_decoder(latent.to(dtype=self._decoder_dtype())).float()
        if logits.shape[1] != 1:
            logits = logits.max(dim=1, keepdim=True).values
        return logits

    def build_condition_from_prior(self, prior_coords: torch.Tensor, prior_conf: torch.Tensor, batch_size: int):
        with torch.no_grad():
            partial_occ = coords_to_batched_occ(
                prior_coords,
                batch_size,
                resolution=64,
                device=self.device,
                dtype=self._encoder_dtype(),
            )
            raw_partial_latent = self.ss_encoder(partial_occ, sample_posterior=False).to(torch.float32)
        latent_mask, confidence = partial_latent_stats(
            prior_coords,
            batch_size,
            weights=prior_conf,
            latent_resolution=raw_partial_latent.shape[-1],
            source_resolution=64,
            device=self.device,
        )
        confidence = confidence.pow(self.known_conf_power).clamp(0.0, 1.0)
        cond_partial_latent = raw_partial_latent * latent_mask * confidence
        cond = self.ss_cond(cond_partial_latent, latent_mask, confidence)
        return raw_partial_latent, cond, latent_mask, confidence

    def build_known_inputs(self, batch):
        targets = batch["x_0"].to(self.device, dtype=torch.float32)
        b = int(targets.shape[0])
        prior_coords = batch["prior_coords"].to(self.device, dtype=torch.long)
        prior_conf = batch["prior_conf"].to(self.device, dtype=torch.float32)
        raw_partial_latent, clean_cond, latent_mask, confidence = self.build_condition_from_prior(
            prior_coords,
            prior_conf,
            b,
        )
        cond = clean_cond
        if random.random() < self.cfg_drop_prob:
            cond = torch.zeros_like(cond)
        stats = {
            "prior_points": float(prior_coords.shape[0] / max(b, 1)),
            "latent_mask_ratio": float(latent_mask.mean().detach().cpu()),
            "latent_conf_mean": float(confidence.mean().detach().cpu()),
        }
        return targets, raw_partial_latent, cond, clean_cond, latent_mask, confidence, stats

    def make_negative_prior(self, prior_coords: torch.Tensor, prior_conf: torch.Tensor, batch_size: int, mode: str):
        mode = mode.strip().lower()
        if prior_coords.numel() == 0:
            return prior_coords, prior_conf
        if mode == "shuffle":
            neg_coords = prior_coords.clone()
            neg_conf = prior_conf.clone()
            if batch_size > 1:
                neg_coords[:, 0] = (neg_coords[:, 0] - 1) % batch_size
            else:
                xyz = neg_coords[:, 1:].clone()
                neg_coords[:, 1] = xyz[:, 1]
                neg_coords[:, 2] = xyz[:, 2]
                neg_coords[:, 3] = xyz[:, 0]
            return neg_coords, neg_conf
        if mode == "random":
            counts = torch.bincount(prior_coords[:, 0].clamp(0, batch_size - 1), minlength=batch_size)
            coords = []
            for batch_idx, count in enumerate(counts.tolist()):
                if count <= 0:
                    continue
                xyz = torch.randint(0, 64, (int(count), 3), device=self.device, dtype=torch.long)
                bcol = torch.full((int(count), 1), int(batch_idx), device=self.device, dtype=torch.long)
                coords.append(torch.cat([bcol, xyz], dim=1))
            if not coords:
                return torch.zeros((0, 4), device=self.device, dtype=torch.long), torch.zeros((0,), device=self.device)
            neg_coords = torch.cat(coords, dim=0)
            neg_conf = torch.ones((neg_coords.shape[0],), device=self.device, dtype=prior_conf.dtype)
            return neg_coords, neg_conf
        raise ValueError(f"unsupported ranking negative mode {mode!r}")

    def ranking_losses(
        self,
        *,
        x_t: torch.Tensor,
        t: float,
        clean_cond: torch.Tensor,
        logits_pos: torch.Tensor,
        target_occ: torch.Tensor,
        prior_coords: torch.Tensor,
        prior_conf: torch.Tensor,
    ):
        zero = x_t.new_tensor(0.0)
        out = {
            "loss_rank_outside": zero,
            "loss_rank_observed": zero,
            "loss_wrong_support": zero,
            "loss_rank_target_support": zero,
            "rank_outside_pos": zero,
            "rank_outside_neg": zero,
            "rank_observed_pos": zero,
            "rank_observed_neg": zero,
            "wrong_support_score": zero,
            "target_support_score": zero,
        }
        if self.ranking_loss_weight <= 0 or not self.ranking_negative_modes:
            return zero, out

        batch_size = int(x_t.shape[0])
        prior_occ = coords_to_batched_occ(prior_coords, batch_size, resolution=64, device=self.device, dtype=torch.float32)
        target_support_weight = (prior_occ * target_occ).clamp(0.0, 1.0)
        outside_weight = (1.0 - target_occ).clamp(0.0, 1.0)
        outside_pos = weighted_channel_mean(F.softplus(logits_pos + self.anti_overfill_margin), outside_weight)
        observed_pos = weighted_channel_mean(logits_pos, prior_occ)
        target_support = weighted_channel_mean(F.softplus(-logits_pos), target_support_weight)
        t_tensor = torch.tensor([1000.0 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)

        losses = []
        accum = {
            key: []
            for key in out.keys()
            if key
            not in {
                "loss_rank_outside",
                "loss_rank_observed",
                "loss_wrong_support",
                "loss_rank_target_support",
                "target_support_score",
            }
        }
        rank_outside_losses = []
        rank_observed_losses = []
        wrong_support_losses = []
        for mode in self.ranking_negative_modes:
            neg_coords, neg_conf = self.make_negative_prior(prior_coords, prior_conf, batch_size, mode)
            _neg_partial, neg_cond, _neg_mask, _neg_confidence = self.build_condition_from_prior(neg_coords, neg_conf, batch_size)
            pred_v_neg = self.ss_flow_model(x_t, t_tensor, neg_cond)
            pred_x0_neg = self.ss_sampler._pred_to_xstart(x_t, t, pred_v_neg)
            logits_neg = self.decode_occ_logits(pred_x0_neg)

            outside_neg = weighted_channel_mean(F.softplus(logits_neg + self.anti_overfill_margin), outside_weight)
            observed_neg = weighted_channel_mean(logits_neg, prior_occ)
            neg_occ = coords_to_batched_occ(neg_coords, batch_size, resolution=64, device=self.device, dtype=torch.float32)
            wrong_support_weight = (neg_occ * (1.0 - target_occ)).clamp(0.0, 1.0)
            wrong_support = weighted_channel_mean(F.softplus(logits_neg + self.anti_overfill_margin), wrong_support_weight)

            rank_outside = F.softplus(self.ranking_margin + outside_pos - outside_neg.detach())
            rank_observed = F.softplus(self.ranking_margin + observed_neg - observed_pos)
            rank_wrong_support = wrong_support
            rank_outside_losses.append(rank_outside)
            rank_observed_losses.append(rank_observed)
            wrong_support_losses.append(rank_wrong_support)
            losses.append(
                self.ranking_outside_weight * rank_outside
                + self.ranking_observed_weight * rank_observed
                + self.ranking_wrong_support_weight * rank_wrong_support
            )
            accum["rank_outside_pos"].append(outside_pos.detach())
            accum["rank_outside_neg"].append(outside_neg.detach())
            accum["rank_observed_pos"].append(observed_pos.detach())
            accum["rank_observed_neg"].append(observed_neg.detach())
            accum["wrong_support_score"].append(wrong_support.detach())

        if not losses:
            return zero, out
        loss_rank = torch.stack(losses).mean() + self.ranking_target_support_weight * target_support
        out["loss_rank_outside"] = torch.stack(rank_outside_losses).mean()
        out["loss_rank_observed"] = torch.stack(rank_observed_losses).mean()
        out["loss_wrong_support"] = torch.stack(wrong_support_losses).mean()
        out["loss_rank_target_support"] = target_support
        out["target_support_score"] = target_support.detach()
        for key, vals in accum.items():
            out[key] = torch.stack(vals).mean() if vals else zero
        return loss_rank, out

    def training_step(self, batch, batch_idx):
        t = torch.rand(1).item()
        targets, partial_latent, cond, clean_cond, latent_mask, confidence, stats = self.build_known_inputs(batch)
        noise = torch.randn_like(targets)
        known_weight = (latent_mask * confidence if self.known_use_confidence else latent_mask).clamp(0.0, 1.0)
        inpaint_target = targets * (1.0 - known_weight) + partial_latent * known_weight
        x_t, gt_v = self.ss_sampler._get_model_gt(inpaint_target, t, noise)
        t_tensor = torch.tensor([1000.0 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        pred_v = self.ss_flow_model(x_t, t_tensor, cond)

        flow_mse = torch.nan_to_num(F.mse_loss(pred_v, gt_v, reduction="none"), nan=0.0, posinf=1e4, neginf=1e4)
        loss_flow = flow_mse.mean()
        loss_known_flow = weighted_channel_mean(flow_mse, known_weight)
        loss_unknown_flow = weighted_channel_mean(flow_mse, 1.0 - known_weight)

        pred_x0 = self.ss_sampler._pred_to_xstart(x_t, t, pred_v)
        x0_mse = torch.nan_to_num(F.mse_loss(pred_x0, inpaint_target, reduction="none"), nan=0.0, posinf=1e4, neginf=1e4)
        loss_known_x0 = weighted_channel_mean(x0_mse, known_weight)

        loss_anti_overfill = pred_x0.new_tensor(0.0)
        outside_logit_mean = pred_x0.new_tensor(0.0)
        target_logit_mean = pred_x0.new_tensor(0.0)
        outside_occ_rate = pred_x0.new_tensor(0.0)
        loss_rank = pred_x0.new_tensor(0.0)
        rank_stats = {}
        if self.anti_overfill_loss_weight > 0 or self.ranking_loss_weight > 0:
            target_coords = batch["target_coords"].to(self.device, dtype=torch.long)
            target_occ = coords_to_batched_occ(
                target_coords,
                int(pred_x0.shape[0]),
                resolution=64,
                device=self.device,
                dtype=torch.float32,
            )
            logits = self.decode_occ_logits(pred_x0)
            if logits.shape[-1] != target_occ.shape[-1]:
                raise ValueError(f"decoder logits resolution {tuple(logits.shape)} does not match target occ {tuple(target_occ.shape)}")
            outside_weight = (1.0 - target_occ).clamp(0.0, 1.0)
            if self.anti_overfill_loss_weight > 0:
                loss_anti_overfill = weighted_channel_mean(F.softplus(logits + self.anti_overfill_margin), outside_weight)
            with torch.no_grad():
                outside_logit_mean = weighted_channel_mean(logits, outside_weight)
                target_logit_mean = weighted_channel_mean(logits, target_occ)
                outside_occ_rate = weighted_channel_mean((logits > 0).to(torch.float32), outside_weight)
            if self.ranking_loss_weight > 0:
                prior_coords = batch["prior_coords"].to(self.device, dtype=torch.long)
                prior_conf = batch["prior_conf"].to(self.device, dtype=torch.float32)
                loss_rank, rank_stats = self.ranking_losses(
                    x_t=x_t,
                    t=t,
                    clean_cond=clean_cond,
                    logits_pos=logits,
                    target_occ=target_occ,
                    prior_coords=prior_coords,
                    prior_conf=prior_conf,
                )

        loss_total = (
            loss_flow
            + self.known_flow_loss_weight * loss_known_flow
            + self.known_x0_loss_weight * loss_known_x0
            + self.anti_overfill_loss_weight * loss_anti_overfill
            + self.ranking_loss_weight * loss_rank
        )

        self.log("train_loss", loss_total, prog_bar=True, sync_dist=True)
        self.log("loss_flow", loss_flow, prog_bar=False, sync_dist=True)
        self.log("loss_known_flow", loss_known_flow, prog_bar=False, sync_dist=True)
        self.log("loss_unknown_flow", loss_unknown_flow, prog_bar=False, sync_dist=True)
        self.log("loss_known_x0", loss_known_x0, prog_bar=False, sync_dist=True)
        self.log("loss_anti_overfill", loss_anti_overfill, prog_bar=False, sync_dist=True)
        self.log("loss_rank", loss_rank, prog_bar=False, sync_dist=True)
        for key, value in rank_stats.items():
            self.log(key, value, prog_bar=False, sync_dist=True)
        self.log("outside_logit_mean", outside_logit_mean, prog_bar=False, sync_dist=True)
        self.log("target_logit_mean", target_logit_mean, prog_bar=False, sync_dist=True)
        self.log("outside_occ_rate", outside_occ_rate, prog_bar=False, sync_dist=True)
        self.log("prior_points", stats["prior_points"], prog_bar=False, sync_dist=True)
        self.log("latent_mask_ratio", stats["latent_mask_ratio"], prog_bar=False, sync_dist=True)
        self.log("latent_conf_mean", stats["latent_conf_mean"], prog_bar=False, sync_dist=True)
        self.log("known_weight_mean", float(known_weight.mean().detach().cpu()), prog_bar=False, sync_dist=True)
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


def build_model(args: argparse.Namespace, local_rank: int) -> SparseInpaintStage2Trainer:
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

    ss_decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    for p in ss_decoder.parameters():
        p.requires_grad = False

    ss_cond = SparsePointPriorCond(
        latent_channels=args.latent_channels,
        cond_channels=args.cond_channels,
        grid_resolution=args.latent_grid_resolution,
    ).to(device).train()
    model = SparseInpaintStage2Trainer(
        ss_flow_model=ss_flow_model,
        ss_cond=ss_cond,
        ss_encoder=ss_encoder,
        ss_decoder=ss_decoder,
        ss_sampler=pipeline.sparse_structure_sampler,
        lr=args.lr,
        cfg_drop_prob=args.cfg_drop_prob,
        known_flow_loss_weight=args.known_flow_loss_weight,
        known_x0_loss_weight=args.known_x0_loss_weight,
        known_conf_power=args.known_conf_power,
        known_use_confidence=args.known_use_confidence,
        anti_overfill_loss_weight=args.anti_overfill_loss_weight,
        anti_overfill_margin=args.anti_overfill_margin,
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_margin=args.ranking_margin,
        ranking_negative_modes=args.ranking_negative_modes,
        ranking_outside_weight=args.ranking_outside_weight,
        ranking_observed_weight=args.ranking_observed_weight,
        ranking_wrong_support_weight=args.ranking_wrong_support_weight,
        ranking_target_support_weight=args.ranking_target_support_weight,
    )
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu")
        missing, unexpected = model.load_state_dict(state.get("state_dict", state), strict=False)
        print(
            f"[train_sparse_inpaint_stage2] loaded {args.resume}: "
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_drop_prob", type=float, default=0.05)
    parser.add_argument("--known_flow_loss_weight", type=float, default=2.0)
    parser.add_argument("--known_x0_loss_weight", type=float, default=1.0)
    parser.add_argument("--known_conf_power", type=float, default=1.0)
    parser.add_argument("--known_use_confidence", action="store_true")
    parser.add_argument("--anti_overfill_loss_weight", type=float, default=0.0)
    parser.add_argument("--anti_overfill_margin", type=float, default=0.0)
    parser.add_argument("--ranking_loss_weight", type=float, default=0.0)
    parser.add_argument("--ranking_margin", type=float, default=0.05)
    parser.add_argument("--ranking_negative_modes", default="shuffle,random")
    parser.add_argument("--ranking_outside_weight", type=float, default=1.0)
    parser.add_argument("--ranking_observed_weight", type=float, default=1.0)
    parser.add_argument("--ranking_wrong_support_weight", type=float, default=1.0)
    parser.add_argument("--ranking_target_support_weight", type=float, default=0.0)
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
    pl.seed_everything(int(args.seed) + local_rank, workers=True)

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
        filename="ss-pointprior-stage2-{epoch:02d}-{step}",
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
