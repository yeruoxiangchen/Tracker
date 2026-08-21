from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline

from .camera import crop_resize_with_intrinsics, ensure_resized_with_intrinsics
from .condition import ARDinoRayCond
from .projected_condition import ARProjectedSparseCond
from .visual_hull import visual_hull_logit_bias


def _pil_list_to_tensors(images: Iterable[Image.Image]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    rgb_tensors = []
    mask_tensors = []
    has_alpha = False
    for image in images:
        rgba = image.convert("RGBA")
        arr = np.asarray(rgba).astype(np.float32) / 255.0
        rgb_tensors.append(torch.from_numpy(arr[..., :3]).permute(2, 0, 1))
        alpha = torch.from_numpy(arr[..., 3:4]).permute(2, 0, 1)
        mask_tensors.append(alpha)
        has_alpha = has_alpha or bool((alpha < 0.999).any())
    masks = torch.stack(mask_tensors, dim=0) if has_alpha else None
    return torch.stack(rgb_tensors, dim=0), masks


def apply_lora_to_ss_flow(ss_flow_model, r: int = 64, alpha: int = 128):
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=["to_q", "to_kv", "to_out", "to_qkv"],
    )
    return get_peft_model(ss_flow_model, lora_cfg)


class TrellisARPoseTo3DPipeline:
    """
    Independent no-VGGT AR-pose TRELLIS pipeline.

    Sparse structure uses DINO patch features plus AR ray/pose features.
    SLAT generation reuses TRELLIS DINO multi-image conditioning.
    """

    def __init__(self, base_pipeline: TrellisImageTo3DPipeline, ss_cond: torch.nn.Module, low_vram: bool = False):
        self.base = base_pipeline
        self.ss_cond = ss_cond
        self.low_vram = low_vram

    @property
    def device(self):
        return self.base.device

    @property
    def sparse_logit_resolution(self) -> int:
        """Output grid resolution of the sparse structure decoder."""
        flow_model = self.base.models["sparse_structure_flow_model"]
        decoder = self.base.models["sparse_structure_decoder"]
        resolution = int(getattr(flow_model, "resolution", 64))
        upsample_count = sum(
            1
            for block in getattr(decoder, "blocks", [])
            if block.__class__.__name__ == "UpsampleBlock3d"
        )
        return resolution * (2**upsample_count)

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str,
        checkpoint_path: Optional[str] = None,
        device: str | torch.device = "cuda",
        low_vram: bool = False,
        use_image_features: bool = True,
        use_pose_features: bool = True,
        cond_fp16: bool = False,
        condition_mode: str = "ray",
        projected_grid_resolution: int = 16,
        projected_min_support: float = 0.5,
        projected_min_support_ratio: float = 0.15,
        projected_grid_transform: str = "identity",
        lora_rank: int = 64,
        lora_alpha: int = 128,
        apply_lora: bool = True,
    ) -> "TrellisARPoseTo3DPipeline":
        base = TrellisImageTo3DPipeline.from_pretrained(weights_path)
        base.to(torch.device(device))
        base.low_vram = low_vram

        if checkpoint_path is not None and apply_lora:
            base.models["sparse_structure_flow_model"] = apply_lora_to_ss_flow(
                base.models["sparse_structure_flow_model"], r=lora_rank, alpha=lora_alpha
            )
            base.sparse_structure_flow_model = base.models["sparse_structure_flow_model"]

        if condition_mode == "projected":
            if not use_pose_features:
                raise ValueError("Projected sparse condition requires pose/extrinsics; do not disable pose features.")
            ss_cond = ARProjectedSparseCond(
                use_image_features=use_image_features,
                use_mask_features=True,
                grid_resolution=projected_grid_resolution,
                min_support_sum=projected_min_support,
                min_support_ratio=projected_min_support_ratio,
                grid_transform=projected_grid_transform,
                use_fp16=cond_fp16,
            ).to(device).eval()
        elif condition_mode == "ray":
            ss_cond = ARDinoRayCond(
                use_image_features=use_image_features,
                use_pose_features=use_pose_features,
                use_fp16=cond_fp16,
            ).to(device).eval()
        else:
            raise ValueError(f"Unsupported condition_mode={condition_mode!r}")
        pipe = cls(base, ss_cond, low_vram=low_vram)
        if checkpoint_path is not None:
            pipe.load_ar_checkpoint(checkpoint_path, strict=False)
        return pipe

    def load_ar_checkpoint(self, checkpoint_path: str, strict: bool = False):
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            siblings = []
            if checkpoint.parent.exists():
                siblings = sorted(checkpoint.parent.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
            hint = ""
            if siblings:
                hint = " Available ckpts: " + ", ".join(str(path) for path in siblings)
            raise FileNotFoundError(
                f"AR checkpoint not found: {checkpoint}. Run training first or pass an existing .ckpt.{hint}"
            )
        state = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        ss_cond_state = {}
        ss_flow_state = {}
        for key, value in state.items():
            if key.startswith("ss_cond."):
                ss_cond_state[key.replace("ss_cond.", "", 1)] = value
            elif key.startswith("ss_flow_model."):
                ss_flow_state[key.replace("ss_flow_model.", "", 1)] = value
        missing, unexpected = self.ss_cond.load_state_dict(ss_cond_state, strict=strict)
        if missing or unexpected:
            print(f"[ARPosePipeline] ss_cond load missing={len(missing)} unexpected={len(unexpected)}")
        if ss_flow_state:
            missing, unexpected = self.base.models["sparse_structure_flow_model"].load_state_dict(
                ss_flow_state, strict=False
            )
            print(f"[ARPosePipeline] ss_flow load missing={len(missing)} unexpected={len(unexpected)}")

    def prepare_inputs(
        self,
        images: torch.Tensor | list[Image.Image],
        intrinsics: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        resolution: int = 518,
        crop_foreground: bool = True,
        no_background: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if isinstance(images, list):
            images, alpha = _pil_list_to_tensors(images)
            if masks is None:
                masks = alpha
        if images.ndim != 4:
            raise ValueError(f"images should be [V,3,H,W], got {tuple(images.shape)}")
        images = images.to(self.device).float()
        intrinsics = intrinsics.to(self.device).float()
        masks = masks.to(self.device).float() if masks is not None else None

        if crop_foreground and masks is not None:
            return crop_resize_with_intrinsics(
                images,
                masks,
                intrinsics,
                resolution=resolution,
                no_background=no_background,
            )
        return ensure_resized_with_intrinsics(images, masks, intrinsics, resolution=resolution)

    @torch.no_grad()
    def encode_ss_condition(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        extrinsics_are_c2w: bool = True,
        camera_forward_sign: float = 1.0,
        reference_relative_pose: bool = True,
    ) -> dict:
        image_cond = self.base.encode_image(images)
        patch_cond = image_cond[:, 5:].reshape(1, images.shape[0], -1, image_cond.shape[-1])
        cond = self.ss_cond(
            patch_cond,
            intrinsics.reshape(1, images.shape[0], 3, 3),
            extrinsics.to(self.device).float().reshape(1, images.shape[0], 4, 4),
            masks=masks.reshape(1, images.shape[0], 1, images.shape[-2], images.shape[-1]) if masks is not None else None,
            image_size=images.shape[-1],
            extrinsics_are_c2w=extrinsics_are_c2w,
            camera_forward_sign=camera_forward_sign,
            reference_relative_pose=reference_relative_pose,
        )
        return {"cond": cond, "neg_cond": torch.zeros_like(cond)}

    @torch.no_grad()
    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict | None = None,
        threshold: float = 0.0,
        min_coords: int = 0,
        fixed_topk: int | None = None,
        logit_prior: torch.Tensor | None = None,
        logit_prior_stats: dict | None = None,
    ) -> torch.Tensor:
        """Sample sparse structure and avoid empty-coordinate crashes during experiments."""
        logits, logit_stats = self.sample_sparse_logits(
            cond,
            num_samples=num_samples,
            sampler_params=sampler_params,
            logit_prior=logit_prior,
            logit_prior_stats=logit_prior_stats,
        )
        coords, select_stats = self.sparse_coords_from_logits(
            logits,
            threshold=threshold,
            min_coords=min_coords,
            fixed_topk=fixed_topk,
        )
        self.last_sparse_stats = {**select_stats, **logit_stats}
        print(
            "[ARPosePipeline] sparse coords="
            f"{self.last_sparse_stats['num_coords']} "
            f"selection={self.last_sparse_stats['selection_mode']} "
            f"threshold={threshold:.4f} "
            f"fixed_topk={self.last_sparse_stats['fixed_topk']} "
            f"logits=[{self.last_sparse_stats['logits_min']:.4f}, "
            f"{self.last_sparse_stats['logits_max']:.4f}] "
            f"topk_fallback={self.last_sparse_stats['used_topk_fallback']}"
        )
        return coords

    @torch.no_grad()
    def sample_sparse_logits(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict | None = None,
        logit_prior: torch.Tensor | None = None,
        logit_prior_stats: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Run sparse flow and decoder once, returning post-prior occupancy logits."""
        sampler_params = sampler_params or {}
        flow_model = self.base.models["sparse_structure_flow_model"]
        decoder = self.base.models["sparse_structure_decoder"]
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso, device=self.device)
        sampler_params = {**self.base.sparse_structure_sampler_params, **sampler_params}

        if getattr(self.base, "low_vram", False):
            flow_model.to(self.device)
        z_s = self.base.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
        ).samples
        if getattr(self.base, "low_vram", False):
            flow_model.cpu()

        if getattr(self.base, "low_vram", False):
            decoder.to(self.device)
        logits = decoder(z_s)
        if logits.ndim != 5:
            raise ValueError(f"Expected sparse decoder logits [B,C,D,H,W], got {tuple(logits.shape)}")
        if logits.shape[1] != 1:
            logits = logits.max(dim=1, keepdim=True).values
        if logit_prior is not None:
            prior = logit_prior.to(device=logits.device, dtype=logits.dtype)
            if prior.ndim != 5:
                raise ValueError(f"logit_prior should be [B,C,D,H,W], got {tuple(prior.shape)}")
            if prior.shape[-3:] != logits.shape[-3:]:
                prior = F.interpolate(
                    prior.float(),
                    size=tuple(logits.shape[-3:]),
                    mode="trilinear",
                    align_corners=False,
                ).to(dtype=logits.dtype)
            if prior.shape[0] == 1 and logits.shape[0] != 1:
                prior = prior.expand(logits.shape[0], -1, -1, -1, -1)
            if prior.shape[1] == 1 and logits.shape[1] != 1:
                prior = prior.expand(-1, logits.shape[1], -1, -1, -1)
            if prior.shape != logits.shape:
                raise ValueError(f"logit_prior shape {tuple(prior.shape)} does not match logits {tuple(logits.shape)}")
            logits = logits + prior

        stats = {
            "logits_min": float(logits.min().detach().cpu()),
            "logits_max": float(logits.max().detach().cpu()),
            "logits_mean": float(logits.mean().detach().cpu()),
        }
        if logit_prior_stats:
            stats.update(logit_prior_stats)
        if getattr(self.base, "low_vram", False):
            decoder.cpu()
            torch.cuda.empty_cache()
        return logits, stats

    @staticmethod
    def sparse_coords_from_logits(
        logits: torch.Tensor,
        *,
        threshold: float = 0.0,
        min_coords: int = 0,
        fixed_topk: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Convert decoder logits to sparse coords.

        fixed_topk selects exactly top-k voxels per sample from logits and bypasses
        thresholding/fallback. This is intended for ranking diagnostics where the
        predicted coordinate count must be controlled across pose modes.
        """
        if logits.ndim != 5:
            raise ValueError(f"Expected sparse decoder logits [B,C,D,H,W], got {tuple(logits.shape)}")

        def topk_coords(k: int) -> tuple[torch.Tensor, int]:
            b, _, d, h, w = logits.shape
            flat = logits.reshape(b, -1)
            per_sample_k = max(0, min(int(k), flat.shape[1]))
            if per_sample_k == 0:
                return torch.zeros((0, 4), device=logits.device, dtype=torch.int32), 0
            _, flat_idx = torch.topk(flat, k=per_sample_k, dim=1)
            spatial = d * h * w
            flat_idx = flat_idx % spatial
            z = flat_idx // (h * w)
            y = (flat_idx % (h * w)) // w
            x = flat_idx % w
            batch_idx = torch.arange(b, device=logits.device, dtype=torch.int64)[:, None].expand_as(x)
            coords_t = torch.stack([batch_idx, z, y, x], dim=-1).reshape(-1, 4).int()
            return coords_t, per_sample_k

        used_topk = False
        selection_mode = "threshold"
        effective_topk = 0
        requested_topk = int(fixed_topk) if fixed_topk is not None else 0
        if fixed_topk is not None:
            if int(fixed_topk) < 0:
                raise ValueError(f"fixed_topk must be >= 0, got {fixed_topk}")
            coords, effective_topk = topk_coords(int(fixed_topk))
            selection_mode = "fixed_topk"
        else:
            coords = torch.argwhere(logits > float(threshold))[:, [0, 2, 3, 4]].int()
            if min_coords > 0 and coords.shape[0] < min_coords:
                coords, effective_topk = topk_coords(int(min_coords))
                used_topk = True
                selection_mode = "threshold_topk_fallback"

        stats = {
            "threshold": float(threshold),
            "min_coords": int(min_coords),
            "fixed_topk": requested_topk,
            "effective_topk": int(effective_topk),
            "selection_mode": selection_mode,
            "num_coords": int(coords.shape[0]),
            "used_topk_fallback": bool(used_topk),
        }
        return coords, stats

    @torch.no_grad()
    def run(
        self,
        images: torch.Tensor | list[Image.Image],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = None,
        slat_sampler_params: dict = None,
        formats: list[str] = None,
        crop_foreground: bool = True,
        no_background: bool = True,
        extrinsics_are_c2w: bool = True,
        camera_forward_sign: float = 1.0,
        reference_relative_pose: bool = True,
        slat_mode: str = "multidiffusion",
        sparse_threshold: float = 0.0,
        min_sparse_coords: int = 0,
        fixed_sparse_topk: int | None = None,
        visual_hull_prior_weight: float = 0.0,
        visual_hull_mask_threshold: float = 0.5,
        visual_hull_min_visible_views: int = 1,
        coords_override: Optional[torch.Tensor] = None,
        coords_override_stats: Optional[dict] = None,
    ):
        sparse_structure_sampler_params = sparse_structure_sampler_params or {}
        slat_sampler_params = slat_sampler_params or {}
        formats = formats or ["mesh", "gaussian"]

        images, masks, intrinsics = self.prepare_inputs(
            images,
            intrinsics,
            masks=masks,
            crop_foreground=crop_foreground,
            no_background=no_background,
        )
        extrinsics = extrinsics.to(self.device).float()

        torch.manual_seed(seed)
        if coords_override is not None:
            coords = coords_override.to(self.device)
            if coords.ndim != 2 or coords.shape[1] not in (3, 4):
                raise ValueError(f"coords_override should be [N,3] or [N,4], got {tuple(coords.shape)}")
            coords = coords.long()
            if coords.shape[1] == 3:
                batch = torch.zeros((coords.shape[0], 1), device=coords.device, dtype=torch.long)
                coords = torch.cat([batch, coords], dim=1)
            coords = coords.int()
            self.last_sparse_stats = {
                "coords_source": "override",
                "num_coords": int(coords.shape[0]),
            }
            if coords_override_stats:
                self.last_sparse_stats.update(coords_override_stats)
            print(f"[ARPosePipeline] sparse coords={coords.shape[0]} source=override")
        else:
            ss_cond = self.encode_ss_condition(
                images,
                intrinsics,
                extrinsics,
                masks=masks,
                extrinsics_are_c2w=extrinsics_are_c2w,
                camera_forward_sign=camera_forward_sign,
                reference_relative_pose=reference_relative_pose,
            )
            logit_prior = None
            logit_prior_stats = None
            if float(visual_hull_prior_weight) != 0.0:
                logit_prior, logit_prior_stats = visual_hull_logit_bias(
                    masks if masks is not None else torch.ones(
                        (images.shape[0], 1, images.shape[-2], images.shape[-1]),
                        device=images.device,
                        dtype=images.dtype,
                    ),
                    intrinsics,
                    extrinsics,
                    extrinsics_are_c2w=extrinsics_are_c2w,
                    resolution=self.sparse_logit_resolution,
                    mask_threshold=visual_hull_mask_threshold,
                    min_visible_views=visual_hull_min_visible_views,
                    weight=visual_hull_prior_weight,
                )
            coords = self.sample_sparse_structure(
                ss_cond,
                num_samples,
                sparse_structure_sampler_params,
                threshold=sparse_threshold,
                min_coords=min_sparse_coords,
                fixed_topk=fixed_sparse_topk,
                logit_prior=logit_prior,
                logit_prior_stats=logit_prior_stats,
            )

        image_cond = self.base.encode_image(images)
        slat_cond = {"cond": image_cond, "neg_cond": torch.zeros_like(image_cond[:1])}
        slat_steps = {**self.base.slat_sampler_params, **slat_sampler_params}.get("steps")
        with self.base.inject_sampler_multi_image("slat_sampler", images.shape[0], slat_steps, mode=slat_mode):
            slat = self.base.sample_slat(slat_cond, coords, slat_sampler_params)
        return self.base.decode_slat(slat, formats), coords
