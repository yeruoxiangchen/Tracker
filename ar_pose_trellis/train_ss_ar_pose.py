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

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(RECONVIAGEN_ROOT))
sys.path.insert(0, str(VGGT_WHEEL_ROOT))

from ar_pose_trellis.camera import crop_resize_with_intrinsics, ensure_resized_with_intrinsics
from ar_pose_trellis.condition import ARDinoRayCond
from ar_pose_trellis.objaverse_pose_dataset import ObjaversePoseDataset, custom_collate as objaverse_pose_collate
from ar_pose_trellis.pipeline import apply_lora_to_ss_flow
from trellis import models as trellis_models
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline


def _target_coords_bad(coords: torch.Tensor, batch_size: int) -> torch.Tensor:
    return (
        (coords[:, 0] < 0)
        | (coords[:, 0] >= batch_size)
        | (coords[:, 1:] < 0).any(dim=1)
        | (coords[:, 1:] >= 64).any(dim=1)
    )


def _reload_target_coords_from_npzs(sample_npzs: list[str], batch_size: int) -> torch.Tensor:
    if len(sample_npzs) != batch_size:
        raise ValueError(f"Cannot repair target_coords: {len(sample_npzs)} npz paths for batch_size={batch_size}")

    repaired = []
    for batch_idx, npz_path in enumerate(sample_npzs):
        with np.load(npz_path) as data:
            coords_np = np.asarray(data["target_coords"])
        if coords_np.ndim != 2 or coords_np.shape[1] not in (3, 4):
            raise ValueError(f"Cannot repair target_coords from {npz_path}: bad shape {coords_np.shape}")
        coords_np = coords_np[:, -3:].astype(np.int64, copy=True)
        bad_np = (
            ~np.isfinite(coords_np).all(axis=1)
            | (coords_np < 0).any(axis=1)
            | (coords_np >= 64).any(axis=1)
        )
        if bad_np.any():
            raise ValueError(f"Cannot repair target_coords from {npz_path}: bad rows {coords_np[bad_np][:10].tolist()}")
        batch_col = np.full((coords_np.shape[0], 1), batch_idx, dtype=np.int64)
        repaired.append(torch.from_numpy(np.concatenate([batch_col, coords_np], axis=1)))
    return torch.cat(repaired, dim=0).contiguous().long()


class ARPoseSSTrainer(pl.LightningModule):
    _dino_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __init__(
        self,
        ss_flow_model: nn.Module,
        ss_cond: nn.Module,
        image_cond_model: nn.Module,
        ss_encoder: nn.Module,
        ss_sampler,
        lr: float = 1e-4,
        cfg_drop_prob: float = 0.1,
        crop_foreground: bool = True,
        extrinsics_are_c2w: bool = True,
        camera_forward_sign: float = 1.0,
        reference_relative_pose: bool = True,
    ):
        super().__init__()
        self.ss_flow_model = ss_flow_model
        self.ss_cond = ss_cond
        self.image_cond_model = image_cond_model
        self.ss_encoder = ss_encoder
        self.ss_sampler = ss_sampler
        self.lr = lr
        self.cfg_drop_prob = cfg_drop_prob
        self.crop_foreground = crop_foreground
        self.extrinsics_are_c2w = extrinsics_are_c2w
        self.camera_forward_sign = camera_forward_sign
        self.reference_relative_pose = reference_relative_pose

    @torch.no_grad()
    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = self._dino_transform(image.to(self.device))
        feats = self.image_cond_model(image, is_training=True)["x_prenorm"]
        return F.layer_norm(feats, feats.shape[-1:])

    def _prepare_batch_views(self, batch):
        images = batch["ref_image"].to(self.device).float()
        alpha = batch["alpha"].to(self.device).float()
        intrinsics = batch["batch_intrinsics"].to(self.device).float()
        extrinsics = batch["batch_extrinsics"].to(self.device).float()
        b, n, c, h, w = images.shape

        images_flat = images.reshape(b * n, c, h, w)
        alpha_flat = alpha.reshape(b * n, 1, h, w)
        intr_flat = intrinsics.reshape(b * n, 3, 3)
        if self.crop_foreground:
            images_flat, alpha_flat, intr_flat = crop_resize_with_intrinsics(
                images_flat, alpha_flat, intr_flat, resolution=518, no_background=True
            )
        else:
            images_flat, alpha_flat, intr_flat = ensure_resized_with_intrinsics(
                images_flat, alpha_flat, intr_flat, resolution=518
            )
        return (
            images_flat.reshape(b, n, 3, 518, 518),
            alpha_flat.reshape(b, n, 1, 518, 518),
            intr_flat.reshape(b, n, 3, 3),
            extrinsics,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Keep integer voxel coordinates on CPU. They are only used to build
        # the occupancy target and do not need Lightning's automatic CUDA
        # transfer. Moving them through the default transfer path caused rare
        # corrupted int64 values in long training runs.
        moved = {}
        for key, value in batch.items():
            if key == "target_coords":
                moved[key] = value.detach().cpu().contiguous().long().clone()
            elif torch.is_tensor(value):
                moved[key] = value.to(device, non_blocking=False)
            else:
                moved[key] = value
        return moved

    def get_input(self, batch):
        target_coords_cpu = batch["target_coords"].detach().to(device="cpu", dtype=torch.long, copy=True).contiguous()
        if target_coords_cpu.ndim != 2 or target_coords_cpu.shape[1] != 4:
            raise ValueError(f"target_coords should be [N,4], got {tuple(target_coords_cpu.shape)}")
        b = int(batch["ref_image"].shape[0])
        bad = _target_coords_bad(target_coords_cpu, b)
        if bad.any():
            bad_rows = target_coords_cpu[bad][:10].tolist()
            print(
                "[train_ss_ar_pose] repairing corrupted batch target_coords: "
                f"bad_rows={bad_rows}, uids={batch.get('sample_uids')}, npzs={batch.get('sample_npzs')}",
                flush=True,
            )
            target_coords_cpu = _reload_target_coords_from_npzs(batch.get("sample_npzs", []), b)
            bad = _target_coords_bad(target_coords_cpu, b)
            if bad.any():
                bad_rows = target_coords_cpu[bad][:10].tolist()
                raise ValueError(
                    "target_coords out of range before GPU condition after repair: "
                    f"bad_rows={bad_rows}, "
                    f"uids={batch.get('sample_uids')}, npzs={batch.get('sample_npzs')}"
                )

        with torch.no_grad():
            # Build the occupancy grid on CPU. CUDA advanced indexing reports
            # out-of-range errors asynchronously and can poison the whole
            # training process; CPU indexing gives deterministic validation.
            ss_cpu = torch.zeros(b, 64, 64, 64, dtype=torch.float32)
            ss_cpu[
                target_coords_cpu[:, 0],
                target_coords_cpu[:, 1],
                target_coords_cpu[:, 2],
                target_coords_cpu[:, 3],
            ] = 1.0
            ss = ss_cpu.to(self.device)
            targets = self.ss_encoder(
                ss.unsqueeze(1).float().to(next(self.ss_encoder.parameters()).dtype),
                sample_posterior=False,
            ).to(torch.float32)

        images_518, alpha_518, intrinsics, extrinsics = self._prepare_batch_views(batch)
        image_batch, n = images_518.shape[:2]
        if image_batch != b:
            raise ValueError(
                "Prepared image batch size changed unexpectedly: "
                f"raw_batch={b}, image_batch={image_batch}, "
                f"uids={batch.get('sample_uids')}, npzs={batch.get('sample_npzs')}"
            )
        if getattr(self.ss_cond, "use_image_features", True):
            with torch.no_grad():
                image_cond = self._encode_image(images_518.reshape(b * n, 3, 518, 518))
                image_patch_cond = image_cond.reshape(b, n, -1, image_cond.shape[-1])[:, :, 5:]
        else:
            patch_embed = getattr(self.image_cond_model, "patch_embed", None)
            patch_size = getattr(patch_embed, "patch_size", 14)
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]
            patch_side = 518 // int(patch_size)
            image_patch_cond = images_518.new_empty((b, n, patch_side * patch_side, 1))
        cond = self.ss_cond(
            image_patch_cond,
            intrinsics,
            extrinsics,
            masks=alpha_518,
            image_size=518,
            extrinsics_are_c2w=self.extrinsics_are_c2w,
            camera_forward_sign=self.camera_forward_sign,
            reference_relative_pose=self.reference_relative_pose,
        )
        if random.random() < self.cfg_drop_prob:
            cond = torch.zeros_like(cond)
        return targets, cond, torch.randn_like(targets)

    def training_step(self, batch, batch_idx):
        t = torch.rand(1).item()
        targets, cond, noise = self.get_input(batch)
        x_t, gt_v = self.ss_sampler._get_model_gt(targets, t, noise)
        t_tensor = torch.tensor([1000.0 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        pred_v = self.ss_flow_model(x_t, t_tensor, cond)
        loss = F.mse_loss(pred_v, gt_v, reduction="none")
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4).mean()
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

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


def build_models(args, local_rank: int):
    device = f"cuda:{local_rank}"
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.to(device)

    ss_flow_model = pipeline.models["sparse_structure_flow_model"].to(device).eval()
    for p in ss_flow_model.parameters():
        p.requires_grad = False
    ss_flow_model = apply_lora_to_ss_flow(ss_flow_model, r=args.lora_rank, alpha=args.lora_alpha)

    image_cond_model = pipeline.models["image_cond_model"].to(device).eval()
    for p in image_cond_model.parameters():
        p.requires_grad = False

    ss_encoder = trellis_models.from_pretrained(
        f"{args.weights}/ckpts/ss_enc_conv3d_16l8_fp16"
        if os.path.isdir(args.weights)
        else f"{args.weights}/ckpts/ss_enc_conv3d_16l8_fp16"
    ).to(device).eval()
    for p in ss_encoder.parameters():
        p.requires_grad = False

    if args.cond_fp16:
        print("[train_ss_ar_pose] Ignoring --cond_fp16 during training; 16-mixed keeps trainable weights fp32.")
    if args.pose_only and args.image_only:
        raise ValueError("--pose_only and --image_only are mutually exclusive.")
    ss_cond = ARDinoRayCond(
        use_image_features=not args.pose_only,
        use_pose_features=not args.image_only,
        use_fp16=False,
    ).to(device).train()
    model = ARPoseSSTrainer(
        ss_flow_model=ss_flow_model,
        ss_cond=ss_cond,
        image_cond_model=image_cond_model,
        ss_encoder=ss_encoder,
        ss_sampler=pipeline.sparse_structure_sampler,
        lr=args.lr,
        cfg_drop_prob=args.cfg_drop_prob,
        crop_foreground=not args.no_crop,
        extrinsics_are_c2w=args.extrinsics_type == "c2w",
        camera_forward_sign=args.camera_forward_sign,
        reference_relative_pose=not args.absolute_pose_condition,
    )
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu")
        missing, unexpected = model.load_state_dict(state.get("state_dict", state), strict=False)
        loaded_keys = len(state.get("state_dict", state))
        print(
            f"[train_ss_ar_pose] loaded trainable weights from {args.resume}: "
            f"keys={loaded_keys}, missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--dataset_format",
        choices=["objaverse_pose", "proobjaverse_tar"],
        default="objaverse_pose",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="Load trainable AR-pose weights from a checkpoint. By default this is not a full Lightning resume.",
    )
    parser.add_argument(
        "--resume_full_state",
        action="store_true",
        help="Use Lightning full-state resume. Only valid for checkpoints that include frozen modules and optimizer state.",
    )
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--accum_batches", type=int, default=1)
    parser.add_argument("--ckpt_every_n_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cfg_drop_prob", type=float, default=0.1)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--pose_only", action="store_true", help="Ablate DINO in SS cond; use only AR ray/mask tokens.")
    parser.add_argument("--image_only", action="store_true", help="Ablate AR ray/mask pose features; use only DINO image tokens.")
    parser.add_argument("--no_crop", action="store_true", help="Resize without foreground crop.")
    parser.add_argument("--cond_fp16", action="store_true")
    parser.add_argument("--extrinsics_type", choices=["c2w", "w2c"], default="c2w")
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument(
        "--absolute_pose_condition",
        action="store_true",
        help="Use absolute input camera poses in the pose condition. Default is reference-relative poses.",
    )
    return parser.parse_args()


def main():
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
    model = build_models(args, local_rank)

    if args.dataset_format == "objaverse_pose":
        dataset = ObjaversePoseDataset(
            args.data_root,
            split=args.split,
            num_views=args.num_views,
            image_size=0,
            random_views=True,
        )
        collate_fn = objaverse_pose_collate
    else:
        from dataset import TarDataset, custom_collate

        dataset = TarDataset(args.data_root, num_views=args.num_views, image_size=0)
        collate_fn = custom_collate
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=world_rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    logger = TensorBoardLogger(args.save_dir, name="tb", version=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    ckpt_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="ss-arpose-{epoch:02d}-{step}",
        save_top_k=-1,
        every_n_train_steps=args.ckpt_every_n_steps,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=world_size if torch.cuda.is_available() else 1,
        strategy="ddp_find_unused_parameters_true" if world_size > 1 else "auto",
        precision="16-mixed",
        logger=logger,
        callbacks=[ckpt_cb],
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        accumulate_grad_batches=args.accum_batches,
        log_every_n_steps=10,
    )
    trainer.fit(model, loader, ckpt_path=args.resume if args.resume_full_state else None)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
