from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from trellis.modules.transformer import ModulatedTransformerCrossBlock_woT
from trellis.modules.utils import convert_module_to_f16

from .camera import build_patch_ray_features


class ARDinoRayCond(nn.Module):
    """No-VGGT sparse-structure condition: DINO patch tokens + AR ray/pose tokens."""

    def __init__(
        self,
        channels: int = 1024,
        dino_channels: int = 1024,
        pose_channels: int = 16,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_init_tokens: int = 4096,
        use_image_features: bool = True,
        use_pose_features: bool = True,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = False,
        dtype: torch.dtype = torch.float32,
        use_fp16: bool = False,
    ):
        super().__init__()
        if not use_image_features and not use_pose_features:
            raise ValueError("ARDinoRayCond needs at least one of image or pose features enabled.")
        self.channels = channels
        self.use_image_features = use_image_features
        self.use_pose_features = use_pose_features
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else dtype

        self.image_proj = nn.Linear(dino_channels, channels) if use_image_features else None
        self.pose_proj = (
            nn.Sequential(
                nn.Linear(pose_channels, channels),
                nn.SiLU(),
                nn.Linear(channels, channels),
            )
            if use_pose_features
            else None
        )
        self.pose_scale = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        self.multiview_cond_tokens = nn.Parameter(torch.randn(1, num_init_tokens, channels, dtype=dtype))
        nn.init.normal_(self.multiview_cond_tokens, std=1e-6)

        self.cond_blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlock_woT(
                    channels,
                    channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_mode="full",
                    use_checkpoint=use_checkpoint,
                    use_rope=False,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(4)
            ]
        )
        if use_fp16:
            self.convert_to_fp16()

    def _forward_dtype(self, device: torch.device) -> torch.dtype:
        if self.use_fp16:
            return torch.float16
        if device.type == "cuda":
            try:
                autocast_enabled = torch.is_autocast_enabled("cuda")
            except TypeError:
                autocast_enabled = torch.is_autocast_enabled()
            if autocast_enabled:
                try:
                    return torch.get_autocast_dtype("cuda")
                except (AttributeError, TypeError):
                    return torch.get_autocast_gpu_dtype()
        return self.dtype

    def convert_to_fp16(self) -> None:
        self.use_fp16 = True
        self.dtype = torch.float16
        self.cond_blocks.apply(convert_module_to_f16)
        if self.pose_proj is not None:
            self.pose_proj.apply(convert_module_to_f16)
        if self.image_proj is not None:
            self.image_proj.apply(convert_module_to_f16)
        self.multiview_cond_tokens = nn.Parameter(self.multiview_cond_tokens.data.to(self.dtype))

    def forward(
        self,
        image_patch_cond: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        image_size: int = 518,
        extrinsics_are_c2w: bool = True,
        camera_forward_sign: float = 1.0,
        reference_relative_pose: bool = True,
        reference_index: int = 0,
    ) -> torch.Tensor:
        if image_patch_cond.ndim != 4:
            raise ValueError(f"image_patch_cond should be [B,V,P,C], got {tuple(image_patch_cond.shape)}")
        b, v, p, _ = image_patch_cond.shape
        patch_side = int(round(p ** 0.5))
        if patch_side * patch_side != p:
            raise ValueError(f"Expected square DINO patch grid, got P={p}")

        forward_dtype = self._forward_dtype(image_patch_cond.device)
        context_terms = []
        if self.image_proj is not None:
            context_terms.append(self.image_proj(image_patch_cond.to(forward_dtype)))
        if self.pose_proj is not None:
            pose_feat = build_patch_ray_features(
                intrinsics=intrinsics.to(image_patch_cond.device),
                extrinsics=extrinsics.to(image_patch_cond.device),
                patch_hw=(patch_side, patch_side),
                image_hw=(image_size, image_size),
                masks=masks.to(image_patch_cond.device) if masks is not None else None,
                extrinsics_are_c2w=extrinsics_are_c2w,
                camera_forward_sign=camera_forward_sign,
                reference_relative=reference_relative_pose,
                reference_index=reference_index,
            )
            pose_ctx = self.pose_proj(pose_feat.to(forward_dtype))
            context_terms.append(self.pose_scale.to(forward_dtype) * pose_ctx)
        if not context_terms:
            raise RuntimeError("ARDinoRayCond has no enabled context terms.")
        context = context_terms[0]
        for term in context_terms[1:]:
            context = context + term

        context = context.reshape(b, v * p, self.channels)
        cond = self.multiview_cond_tokens.repeat(b, 1, 1).to(forward_dtype)
        for block in self.cond_blocks:
            cond = block(cond, context)
        return cond
