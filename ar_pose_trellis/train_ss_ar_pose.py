from __future__ import annotations

import argparse
import datetime
import json
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
from PIL import Image
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
from ar_pose_trellis.projected_condition import ARProjectedSparseCond
from trellis import models as trellis_models
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline


def _resolve_path(root: str | None, path: str) -> str:
    if os.path.isabs(path) or not root:
        return path
    return os.path.join(root, path)


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


class Pixal3DMultiviewSparseManifestDataset(torch.utils.data.Dataset):
    """Reader for pixal3d_multiview manifests with precomputed TRELLIS SS latents."""

    def __init__(
        self,
        manifest: str,
        *,
        max_frames: int = 8,
        apply_mask: bool = True,
    ):
        self.manifest_path = manifest
        self.max_frames = int(max_frames)
        self.apply_mask = bool(apply_mask)
        with open(manifest, "r") as f:
            data = json.load(f)
        samples = data.get("samples", data.get("instances", data.get("framesets", None)))
        if samples is None and isinstance(data, list):
            samples = data
        if samples is None:
            raise ValueError(f"pixal3d_multiview manifest has no samples list: {manifest}")
        self.samples = samples
        self.default_image_root = data.get("image_root")
        self.default_mask_root = data.get("mask_root")
        self.default_latent_root = data.get("latent_root")
        self.default_extrinsics_type = data.get("extrinsics_type", "c2w")
        print(f"[Pixal3DMultiviewSparseManifestDataset] {len(self.samples)} samples from {manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        uid = str(sample.get("uid", sample.get("id", index)))
        frames = sample.get("frames")
        if not frames:
            raise ValueError(f"sample {uid} has no frames")
        frames = frames[: self.max_frames] if self.max_frames > 0 else frames

        image_root = sample.get("image_root", self.default_image_root)
        mask_root = sample.get("mask_root", self.default_mask_root)
        latent_root = sample.get("latent_root", self.default_latent_root)
        top_intrinsic = sample.get("intrinsic")
        extrinsics_type = sample.get("extrinsics_type", self.default_extrinsics_type)

        images = []
        masks = []
        intrinsics = []
        extrinsics = []
        for frame in frames:
            image = Image.open(_resolve_path(image_root, frame["image"])).convert("RGB")
            rgb = np.asarray(image).astype(np.float32) / 255.0
            height, width = rgb.shape[:2]
            mask_path = frame.get("mask", sample.get("mask"))
            if mask_path is not None:
                mask = Image.open(_resolve_path(mask_root, mask_path)).convert("L")
                if mask.size != image.size:
                    mask = mask.resize(image.size, Image.NEAREST)
                mask_arr = np.asarray(mask).astype(np.float32) / 255.0
            else:
                mask_arr = np.ones((height, width), dtype=np.float32)
            if self.apply_mask:
                rgb = rgb * mask_arr[..., None]
            intrinsic = frame.get("intrinsic", top_intrinsic)
            if intrinsic is None:
                raise ValueError(f"missing intrinsic for sample={uid} frame={frame.get('image')}")
            images.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
            masks.append(torch.from_numpy(mask_arr[None]).contiguous())
            intrinsics.append(torch.tensor(intrinsic, dtype=torch.float32))
            extrinsics.append(torch.tensor(frame["extrinsic"], dtype=torch.float32))

        latent_rel = sample.get("ss_latent", sample.get("ss_latent_path", sample.get("latent")))
        if latent_rel is None:
            raise ValueError(f"sample {uid} has no ss_latent/ss_latent_path")
        latent_path = _resolve_path(latent_root, latent_rel)
        with np.load(latent_path) as latent:
            if "z" not in latent:
                raise ValueError(f"sparse latent npz has no key 'z': {latent_path}")
            x_0 = torch.from_numpy(latent["z"].astype(np.float32))
            target_coords = torch.from_numpy(latent["target_coords"].astype(np.int64))
        if x_0.ndim == 5 and x_0.shape[0] == 1:
            x_0 = x_0[0]
        if x_0.ndim != 4:
            raise ValueError(f"expected sparse latent z [C,D,H,W], got {tuple(x_0.shape)} for {latent_path}")
        if target_coords.ndim != 2 or target_coords.shape[1] not in (3, 4):
            raise ValueError(f"expected target_coords [N,3/4], got {tuple(target_coords.shape)} for {latent_path}")
        target_coords = target_coords[:, -3:].contiguous().long()

        return {
            "ref_image": torch.stack(images, dim=0),
            "alpha": torch.stack(masks, dim=0),
            "batch_intrinsics": torch.stack(intrinsics, dim=0),
            "batch_extrinsics": torch.stack(extrinsics, dim=0),
            "x_0": x_0.contiguous(),
            "target_coords": target_coords,
            "sample_uid": uid,
            "sample_npz": latent_path,
            "extrinsics_type": extrinsics_type,
        }


def pixal3d_multiview_collate(batch: list[dict]) -> dict:
    batched_coords = []
    for batch_idx, item in enumerate(batch):
        coords = item["target_coords"].contiguous().long()
        batch_col = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.long)
        batched_coords.append(torch.cat([batch_col, coords[:, -3:]], dim=1))
    return {
        "ref_image": torch.stack([item["ref_image"] for item in batch], dim=0),
        "alpha": torch.stack([item["alpha"] for item in batch], dim=0),
        "batch_intrinsics": torch.stack([item["batch_intrinsics"] for item in batch], dim=0),
        "batch_extrinsics": torch.stack([item["batch_extrinsics"] for item in batch], dim=0),
        "x_0": torch.stack([item["x_0"] for item in batch], dim=0),
        "target_coords": torch.cat(batched_coords, dim=0),
        "sample_uids": [item.get("sample_uid", str(i)) for i, item in enumerate(batch)],
        "sample_npzs": [item.get("sample_npz", "") for item in batch],
        "extrinsics_type": [item.get("extrinsics_type", "c2w") for item in batch],
    }


def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = F.normalize(axis.float(), dim=0)
    x, y, z = axis
    c = torch.cos(angle.float())
    s = torch.sin(angle.float())
    one = torch.ones_like(c)
    zero = torch.zeros_like(c)
    k = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    eye = torch.eye(3, device=axis.device, dtype=torch.float32)
    return c * eye + (one - c) * (axis[:, None] * axis[None, :]) + s * k


def _perturb_c2w(c2w: torch.Tensor, *, max_rot_deg: float, trans_scale: float) -> torch.Tensor:
    out = c2w.clone()
    flat = out.reshape(-1, 4, 4)
    for item_idx in range(flat.shape[0]):
        axis = torch.randn(3, device=c2w.device, dtype=torch.float32)
        angle = (torch.rand((), device=c2w.device) * 2.0 - 1.0) * max_rot_deg * torch.pi / 180.0
        rot = _axis_angle_to_matrix(axis, angle).to(dtype=c2w.dtype, device=c2w.device)
        trans = (torch.rand(3, device=c2w.device, dtype=c2w.dtype) * 2.0 - 1.0) * float(trans_scale)
        flat[item_idx, :3, :3] = rot @ flat[item_idx, :3, :3]
        flat[item_idx, :3, 3] = flat[item_idx, :3, 3] + trans
    return out


def _apply_wrong_pose_mode(extrinsics: torch.Tensor, mode: str, *, extrinsics_are_c2w: bool) -> torch.Tensor:
    mode = mode.strip().lower()
    if mode == "identity":
        eye = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype)
        return eye.reshape(1, 1, 4, 4).expand_as(extrinsics).clone()
    if mode == "shuffle":
        out = extrinsics.clone()
        for batch_idx in range(out.shape[0]):
            perm = torch.randperm(out.shape[1], device=out.device)
            if torch.equal(perm, torch.arange(out.shape[1], device=out.device)):
                perm = torch.roll(perm, shifts=1)
            out[batch_idx] = out[batch_idx, perm]
        return out
    if mode == "reverse":
        return torch.flip(extrinsics, dims=[1])
    if mode in {"cyclic_shift1", "cyclic_shift2"}:
        shift = 1 if mode == "cyclic_shift1" else 2
        return torch.roll(extrinsics, shifts=shift, dims=1)
    if mode in {"noise", "large_noise"}:
        c2w = extrinsics if extrinsics_are_c2w else torch.linalg.inv(extrinsics)
        c2w = _perturb_c2w(
            c2w,
            max_rot_deg=35.0 if mode == "noise" else 90.0,
            trans_scale=0.25 if mode == "noise" else 0.75,
        )
        return c2w if extrinsics_are_c2w else torch.linalg.inv(c2w)
    raise ValueError(f"unknown wrong pose mode: {mode}")


class ARPoseSSTrainer(pl.LightningModule):
    _dino_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __init__(
        self,
        ss_flow_model: nn.Module,
        ss_cond: nn.Module,
        image_cond_model: nn.Module,
        ss_encoder: nn.Module,
        ss_decoder: nn.Module,
        ss_sampler,
        lr: float = 1e-4,
        cfg_drop_prob: float = 0.1,
        crop_foreground: bool = True,
        extrinsics_are_c2w: bool = True,
        camera_forward_sign: float = 1.0,
        reference_relative_pose: bool = True,
        ranking_weight: float = 0.0,
        ranking_margin: float = 0.05,
        ranking_modes: str = "identity,shuffle,large_noise,noise",
        ranking_num_negatives: int = 1,
        ranking_background_samples: int = 4096,
        ranking_every_n_steps: int = 1,
    ):
        super().__init__()
        self.ss_flow_model = ss_flow_model
        self.ss_cond = ss_cond
        self.image_cond_model = image_cond_model
        self.ss_encoder = ss_encoder
        self.ss_decoder = ss_decoder
        self.ss_sampler = ss_sampler
        self.lr = lr
        self.cfg_drop_prob = cfg_drop_prob
        self.crop_foreground = crop_foreground
        self.extrinsics_are_c2w = extrinsics_are_c2w
        self.camera_forward_sign = camera_forward_sign
        self.reference_relative_pose = reference_relative_pose
        self.ranking_weight = float(ranking_weight)
        self.ranking_margin = float(ranking_margin)
        self.ranking_modes = [mode.strip() for mode in str(ranking_modes).split(",") if mode.strip()]
        self.ranking_num_negatives = int(ranking_num_negatives)
        self.ranking_background_samples = int(ranking_background_samples)
        self.ranking_every_n_steps = max(1, int(ranking_every_n_steps))

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

    def _encode_condition_from_prepared(
        self,
        image_patch_cond: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        alpha_518: torch.Tensor,
    ) -> torch.Tensor:
        return self.ss_cond(
            image_patch_cond,
            intrinsics,
            extrinsics,
            masks=alpha_518,
            image_size=518,
            extrinsics_are_c2w=self.extrinsics_are_c2w,
            camera_forward_sign=self.camera_forward_sign,
            reference_relative_pose=self.reference_relative_pose,
        )

    def get_input(self, batch, *, return_aux: bool = False):
        b = int(batch["ref_image"].shape[0])
        if "x_0" in batch:
            targets = batch["x_0"].to(self.device, dtype=torch.float32)
            if targets.ndim != 5:
                raise ValueError(f"x_0 should be [B,C,D,H,W], got {tuple(targets.shape)}")
        else:
            target_coords_cpu = batch["target_coords"].detach().to(device="cpu", dtype=torch.long, copy=True).contiguous()
            if target_coords_cpu.ndim != 2 or target_coords_cpu.shape[1] != 4:
                raise ValueError(f"target_coords should be [N,4], got {tuple(target_coords_cpu.shape)}")
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
        cond = self._encode_condition_from_prepared(
            image_patch_cond,
            intrinsics,
            extrinsics,
            alpha_518,
        )
        cond_for_flow = cond
        if random.random() < self.cfg_drop_prob:
            cond_for_flow = torch.zeros_like(cond)
        aux = {
            "cond_full": cond,
            "image_patch_cond": image_patch_cond,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "alpha_518": alpha_518,
        }
        if "target_coords" in batch:
            aux["target_coords"] = batch["target_coords"].to(self.device, dtype=torch.long)
        if return_aux:
            return targets, cond_for_flow, torch.randn_like(targets), aux
        return targets, cond_for_flow, torch.randn_like(targets)

    def _decode_sparse_logits(self, x_0_pred: torch.Tensor) -> torch.Tensor:
        logits = self.ss_decoder(x_0_pred)
        if logits.ndim != 5:
            raise ValueError(f"Expected sparse decoder logits [B,C,D,H,W], got {tuple(logits.shape)}")
        if logits.shape[1] != 1:
            logits = logits.max(dim=1, keepdim=True).values
        return logits.float()

    def _ranking_score(self, logits: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        if target_coords.ndim != 2 or target_coords.shape[1] != 4:
            raise ValueError(f"target_coords should be [N,4], got {tuple(target_coords.shape)}")
        logits_grid = logits[:, 0]
        b, d, h, w = logits_grid.shape
        coords = target_coords.long()
        valid = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < b)
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < d)
            & (coords[:, 2] >= 0)
            & (coords[:, 2] < h)
            & (coords[:, 3] >= 0)
            & (coords[:, 3] < w)
        )
        coords = coords[valid]
        scores = []
        for batch_idx in range(b):
            coords_b = coords[coords[:, 0] == batch_idx]
            if coords_b.numel() == 0:
                target_mean = logits_grid[batch_idx].mean()
            else:
                target_values = logits_grid[
                    coords_b[:, 0],
                    coords_b[:, 1],
                    coords_b[:, 2],
                    coords_b[:, 3],
                ]
                target_mean = target_values.mean()
            bg_count = max(1, int(self.ranking_background_samples))
            rand_xyz = torch.stack(
                [
                    torch.randint(0, d, (bg_count,), device=logits.device),
                    torch.randint(0, h, (bg_count,), device=logits.device),
                    torch.randint(0, w, (bg_count,), device=logits.device),
                ],
                dim=1,
            )
            bg_values = logits_grid[
                torch.full((bg_count,), batch_idx, device=logits.device, dtype=torch.long),
                rand_xyz[:, 0],
                rand_xyz[:, 1],
                rand_xyz[:, 2],
            ]
            scores.append(target_mean - bg_values.mean())
        return torch.stack(scores).mean()

    def _ranking_loss(
        self,
        aux: dict,
        x_t: torch.Tensor,
        t: float,
        t_tensor: torch.Tensor,
        pred_v_correct: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        target_coords = aux.get("target_coords")
        if target_coords is None:
            zero = pred_v_correct.new_zeros(())
            return zero, {"ranking_skipped": 1.0}
        if not self.ranking_modes or self.ranking_weight <= 0.0:
            zero = pred_v_correct.new_zeros(())
            return zero, {"ranking_skipped": 1.0}

        x0_correct = self.ss_sampler._pred_to_xstart(x_t, t, pred_v_correct)
        logits_correct = self._decode_sparse_logits(x0_correct)
        score_correct = self._ranking_score(logits_correct, target_coords)

        losses = []
        score_wrong_values = []
        chosen_modes = random.choices(self.ranking_modes, k=max(1, self.ranking_num_negatives))
        for mode in chosen_modes:
            wrong_extrinsics = _apply_wrong_pose_mode(
                aux["extrinsics"],
                mode,
                extrinsics_are_c2w=self.extrinsics_are_c2w,
            )
            cond_wrong = self._encode_condition_from_prepared(
                aux["image_patch_cond"],
                aux["intrinsics"],
                wrong_extrinsics,
                aux["alpha_518"],
            )
            pred_v_wrong = self.ss_flow_model(x_t, t_tensor, cond_wrong)
            x0_wrong = self.ss_sampler._pred_to_xstart(x_t, t, pred_v_wrong)
            logits_wrong = self._decode_sparse_logits(x0_wrong)
            score_wrong = self._ranking_score(logits_wrong, target_coords)
            score_wrong_values.append(score_wrong.detach())
            losses.append(F.relu(self.ranking_margin - score_correct + score_wrong))

        if not losses:
            zero = pred_v_correct.new_zeros(())
            return zero, {"ranking_skipped": 1.0}
        loss_rank = torch.stack(losses).mean()
        stats = {
            "ranking_skipped": 0.0,
            "ranking_score_correct": float(score_correct.detach().cpu()),
            "ranking_score_wrong": float(torch.stack(score_wrong_values).mean().cpu()),
        }
        return loss_rank, stats

    def training_step(self, batch, batch_idx):
        t = torch.rand(1).item()
        use_ranking = self.ranking_weight > 0.0 and (self.global_step % self.ranking_every_n_steps == 0)
        targets, cond, noise, aux = self.get_input(batch, return_aux=True)
        x_t, gt_v = self.ss_sampler._get_model_gt(targets, t, noise)
        t_tensor = torch.tensor([1000.0 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        pred_v = self.ss_flow_model(x_t, t_tensor, cond)
        loss = F.mse_loss(pred_v, gt_v, reduction="none")
        loss_flow = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4).mean()
        loss_total = loss_flow
        if use_ranking:
            loss_rank, rank_stats = self._ranking_loss(aux, x_t, t, t_tensor, pred_v)
            loss_rank = torch.nan_to_num(loss_rank, nan=0.0, posinf=1e4, neginf=1e4)
            loss_total = loss_total + self.ranking_weight * loss_rank
            self.log("train_rank_loss", loss_rank, prog_bar=True, sync_dist=True)
            self.log("train_rank_score_correct", rank_stats.get("ranking_score_correct", 0.0), sync_dist=True)
            self.log("train_rank_score_wrong", rank_stats.get("ranking_score_wrong", 0.0), sync_dist=True)
        self.log("train_loss", loss_total, prog_bar=True, sync_dist=True)
        self.log("train_flow_loss", loss_flow, prog_bar=False, sync_dist=True)
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

    ss_decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    for p in ss_decoder.parameters():
        p.requires_grad = False

    if args.cond_fp16:
        print("[train_ss_ar_pose] Ignoring --cond_fp16 during training; 16-mixed keeps trainable weights fp32.")
    if args.pose_only and args.image_only:
        raise ValueError("--pose_only and --image_only are mutually exclusive.")
    if args.condition_mode == "projected":
        if args.image_only:
            raise ValueError("--image_only is not meaningful for --condition_mode projected.")
        ss_cond = ARProjectedSparseCond(
            use_image_features=not args.pose_only,
            use_mask_features=True,
            grid_resolution=args.projected_grid_resolution,
            min_support_sum=args.projected_min_support,
            min_support_ratio=args.projected_min_support_ratio,
            grid_transform=args.projected_grid_transform,
            use_fp16=False,
        ).to(device).train()
    else:
        ss_cond = ARDinoRayCond(
            use_image_features=not args.pose_only,
            use_pose_features=not args.image_only,
            use_fp16=False,
        ).to(device).train()
    if local_rank == 0:
        print(
            f"[train_ss_ar_pose] condition_mode={args.condition_mode} "
            f"cond={ss_cond.__class__.__name__}",
            flush=True,
        )
    model = ARPoseSSTrainer(
        ss_flow_model=ss_flow_model,
        ss_cond=ss_cond,
        image_cond_model=image_cond_model,
        ss_encoder=ss_encoder,
        ss_decoder=ss_decoder,
        ss_sampler=pipeline.sparse_structure_sampler,
        lr=args.lr,
        cfg_drop_prob=args.cfg_drop_prob,
        crop_foreground=not args.no_crop,
        extrinsics_are_c2w=args.extrinsics_type == "c2w",
        camera_forward_sign=args.camera_forward_sign,
        reference_relative_pose=not args.absolute_pose_condition,
        ranking_weight=args.ranking_weight,
        ranking_margin=args.ranking_margin,
        ranking_modes=args.ranking_modes,
        ranking_num_negatives=args.ranking_num_negatives,
        ranking_background_samples=args.ranking_background_samples,
        ranking_every_n_steps=args.ranking_every_n_steps,
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
        choices=["objaverse_pose", "proobjaverse_tar", "pixal3d_multiview_manifest"],
        default="objaverse_pose",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Explicit manifest path for --dataset_format pixal3d_multiview_manifest. "
        "Defaults to DATA_ROOT/SPLIT.json unless --data_root itself is a json file.",
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
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--accum_batches", type=int, default=1)
    parser.add_argument("--ckpt_every_n_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cfg_drop_prob", type=float, default=0.1)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument(
        "--condition_mode",
        choices=["ray", "projected"],
        default="ray",
        help="ray keeps the original ARDinoRayCond; projected uses TRELLIS sparse-grid projected observations.",
    )
    parser.add_argument(
        "--projected_grid_resolution",
        type=int,
        default=16,
        help="Sparse condition grid resolution for --condition_mode projected. 16 gives 4096 TRELLIS SS tokens.",
    )
    parser.add_argument(
        "--projected_min_support",
        type=float,
        default=0.5,
        help="Minimum summed mask support across views before projected appearance features are enabled.",
    )
    parser.add_argument(
        "--projected_min_support_ratio",
        type=float,
        default=0.15,
        help="Minimum support/visible ratio before projected appearance features are enabled.",
    )
    parser.add_argument(
        "--projected_grid_transform",
        choices=["identity", "pixal3d_rotation"],
        default="identity",
        help="Transform sparse latent grid centers before camera projection.",
    )
    parser.add_argument("--pose_only", action="store_true", help="Ablate DINO in SS cond; use only AR ray/mask tokens.")
    parser.add_argument("--image_only", action="store_true", help="Ablate AR ray/mask pose features; use only DINO image tokens.")
    parser.add_argument("--no_crop", action="store_true", help="Resize without foreground crop.")
    parser.add_argument("--no_apply_mask", action="store_true", help="Do not multiply RGB by mask for manifest datasets.")
    parser.add_argument("--cond_fp16", action="store_true")
    parser.add_argument("--extrinsics_type", choices=["c2w", "w2c"], default="c2w")
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument(
        "--absolute_pose_condition",
        action="store_true",
        help="Use absolute input camera poses in the pose condition. Default is reference-relative poses.",
    )
    parser.add_argument(
        "--ranking_weight",
        type=float,
        default=0.0,
        help="Weight for wrong-pose ranking loss. 0 disables ranking and keeps pure flow matching.",
    )
    parser.add_argument("--ranking_margin", type=float, default=0.05)
    parser.add_argument("--ranking_modes", default="identity,shuffle,large_noise,noise")
    parser.add_argument("--ranking_num_negatives", type=int, default=1)
    parser.add_argument("--ranking_background_samples", type=int, default=4096)
    parser.add_argument(
        "--ranking_every_n_steps",
        type=int,
        default=1,
        help="Compute ranking loss every N training steps to reduce overhead.",
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
    elif args.dataset_format == "pixal3d_multiview_manifest":
        if args.manifest is not None:
            manifest = args.manifest
        elif str(args.data_root).endswith(".json"):
            manifest = args.data_root
        else:
            manifest = os.path.join(args.data_root, f"{args.split}.json")
        dataset = Pixal3DMultiviewSparseManifestDataset(
            manifest,
            max_frames=args.num_views,
            apply_mask=not args.no_apply_mask,
        )
        collate_fn = pixal3d_multiview_collate
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
