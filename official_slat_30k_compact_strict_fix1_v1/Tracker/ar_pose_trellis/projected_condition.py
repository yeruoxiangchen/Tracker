from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


PIXAL3D_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)


def _grid_centers(resolution: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords_1d = torch.arange(resolution, device=device, dtype=dtype)
    gx, gy, gz = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    return (coords + 0.5) / float(resolution) - 0.5


class ARProjectedSparseCond(nn.Module):
    """TRELLIS sparse-grid condition from AR-pose multiview projected observations.

    This produces one condition token per TRELLIS sparse latent grid point. Camera
    poses are used to project canonical grid centers into each view, then DINO
    patch features and mask support are aggregated back onto the 3D grid.
    """

    def __init__(
        self,
        channels: int = 1024,
        dino_channels: int = 1024,
        grid_resolution: int = 16,
        use_image_features: bool = True,
        use_mask_features: bool = True,
        min_support_sum: float = 0.5,
        min_support_ratio: float = 0.15,
        grid_transform: str = "identity",
        use_fp16: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.channels = channels
        self.grid_resolution = int(grid_resolution)
        self.use_image_features = bool(use_image_features)
        self.use_mask_features = bool(use_mask_features)
        self.min_support_sum = float(min_support_sum)
        self.min_support_ratio = float(min_support_ratio)
        self.grid_transform = str(grid_transform)
        self.use_fp16 = bool(use_fp16)
        self.dtype = torch.float16 if use_fp16 else dtype
        self.num_tokens = self.grid_resolution**3
        if self.grid_transform not in {"identity", "pixal3d_rotation"}:
            raise ValueError(f"Unsupported grid_transform={self.grid_transform!r}")

        if not self.use_image_features and not self.use_mask_features:
            raise ValueError("ARProjectedSparseCond needs image features or mask/geometric features.")

        self.visible_image_proj = nn.Linear(dino_channels, channels, bias=False) if self.use_image_features else None
        self.support_image_proj = nn.Linear(dino_channels, channels, bias=False) if self.use_image_features else None
        self.geom_proj = nn.Sequential(
            nn.Linear(10, channels, bias=False),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=False),
        )
        self.grid_tokens = nn.Parameter(torch.randn(1, self.num_tokens, channels, dtype=dtype))
        self.support_scale = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        self.geom_scale = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        self.out_norm = nn.LayerNorm(channels)
        nn.init.normal_(self.grid_tokens, std=1e-6)
        if use_fp16:
            self.convert_to_fp16()

    def _transform_grid_centers(self, centers: torch.Tensor) -> torch.Tensor:
        if self.grid_transform == "identity":
            return centers
        if self.grid_transform == "pixal3d_rotation":
            rot = torch.tensor(PIXAL3D_ROTATION, device=centers.device, dtype=centers.dtype)
            return centers @ rot.T
        raise RuntimeError(f"Unsupported grid_transform={self.grid_transform!r}")

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
        self.visible_image_proj = self.visible_image_proj.half() if self.visible_image_proj is not None else None
        self.support_image_proj = self.support_image_proj.half() if self.support_image_proj is not None else None
        self.geom_proj = self.geom_proj.half()
        self.out_norm = self.out_norm.half()
        self.grid_tokens = nn.Parameter(self.grid_tokens.data.half())

    def _project_points(
        self,
        centers: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        *,
        image_size: int,
        extrinsics_are_c2w: bool,
        camera_forward_sign: float,
        reference_relative_pose: bool,
        reference_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, v = intrinsics.shape[:2]
        if reference_index < 0 or reference_index >= v:
            raise ValueError(f"reference_index={reference_index} is out of range for {v} views")
        if abs(float(camera_forward_sign)) < 1e-6:
            raise ValueError("camera_forward_sign must be non-zero.")
        dtype = torch.float32
        centers_f = centers.to(dtype=dtype)
        intrinsics_f = intrinsics.to(dtype=dtype)
        extrinsics_f = extrinsics.to(dtype=dtype)

        ones = torch.ones((centers_f.shape[0], 1), device=centers_f.device, dtype=dtype)
        pts_h = torch.cat([centers_f, ones], dim=1).unsqueeze(0).expand(b, -1, -1)

        c2w = extrinsics_f if extrinsics_are_c2w else torch.linalg.inv(extrinsics_f)
        if reference_relative_pose:
            ref_c2w = c2w[:, reference_index : reference_index + 1]
            w2ref = torch.linalg.inv(ref_c2w)
            c2w = torch.matmul(w2ref, c2w)
            pts_h = torch.einsum("bij,bnj->bni", w2ref[:, 0], pts_h)
        w2c = torch.linalg.inv(c2w)

        cam = torch.einsum("bvij,bnj->bvni", w2c, pts_h)[..., :3]
        signed_depth = cam[..., 2] * float(camera_forward_sign)
        valid_depth = signed_depth > 1e-6
        z = signed_depth.clamp_min(1e-6)
        u = intrinsics_f[..., 0, 0].unsqueeze(-1) * (cam[..., 0] / z) + intrinsics_f[..., 0, 2].unsqueeze(-1)
        vv = intrinsics_f[..., 1, 1].unsqueeze(-1) * (cam[..., 1] / z) + intrinsics_f[..., 1, 2].unsqueeze(-1)

        width = float(image_size)
        height = float(image_size)
        in_image = valid_depth & (u >= 0.0) & (u <= width - 1.0) & (vv >= 0.0) & (vv <= height - 1.0)
        grid_x = 2.0 * (u + 0.5) / width - 1.0
        grid_y = 2.0 * (vv + 0.5) / height - 1.0
        sample_grid = torch.stack([grid_x, grid_y], dim=-1).reshape(b * v, centers.shape[0], 1, 2)
        return sample_grid, in_image, signed_depth

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
        b, v, p, c = image_patch_cond.shape
        patch_side = int(round(p**0.5))
        if patch_side * patch_side != p:
            raise ValueError(f"Expected square DINO patch grid, got P={p}")
        if intrinsics.shape[:2] != (b, v) or extrinsics.shape[:2] != (b, v):
            raise ValueError(
                "view count mismatch: "
                f"patch={tuple(image_patch_cond.shape)} intrinsics={tuple(intrinsics.shape)} "
                f"extrinsics={tuple(extrinsics.shape)}"
            )

        device = image_patch_cond.device
        centers = _grid_centers(self.grid_resolution, device=device, dtype=torch.float32)
        project_centers = self._transform_grid_centers(centers)
        sample_grid, visible, depth = self._project_points(
            project_centers,
            intrinsics.to(device),
            extrinsics.to(device),
            image_size=image_size,
            extrinsics_are_c2w=extrinsics_are_c2w,
            camera_forward_sign=camera_forward_sign,
            reference_relative_pose=reference_relative_pose,
            reference_index=reference_index,
        )
        visible_w = visible.to(dtype=torch.float32)

        mask_values = visible_w
        if masks is not None and self.use_mask_features:
            masks_flat = masks.to(device=device, dtype=torch.float32).reshape(b * v, 1, masks.shape[-2], masks.shape[-1])
            mask_values = F.grid_sample(
                masks_flat,
                sample_grid.to(dtype=masks_flat.dtype),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).reshape(b, v, self.num_tokens).clamp(0.0, 1.0)
            mask_values = mask_values * visible_w

        visible_count = visible_w.sum(dim=1)
        support_sum = mask_values.sum(dim=1)
        visible_denom = visible_count.clamp_min(1.0)
        support_denom = support_sum.clamp_min(1e-4)
        support_ratio = support_sum / visible_denom
        has_visible = (visible_count > 0).to(torch.float32)[..., None]
        has_support = (
            (support_sum >= self.min_support_sum)
            & (support_ratio >= self.min_support_ratio)
        ).to(torch.float32)[..., None]

        cond_terms = [self.grid_tokens.repeat(b, 1, 1).to(self._forward_dtype(device))]
        if self.use_image_features:
            feat_maps = image_patch_cond.reshape(b * v, patch_side, patch_side, c).permute(0, 3, 1, 2).contiguous()
            sampled = F.grid_sample(
                feat_maps,
                sample_grid.to(dtype=feat_maps.dtype),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, v, self.num_tokens, c)
            sampled_f = sampled.to(torch.float32)
            foreground_sum = (sampled_f * mask_values[..., None]).sum(dim=1)
            visible_mean = foreground_sum / visible_denom[..., None]
            support_mean = foreground_sum / support_denom[..., None]
            forward_dtype = self._forward_dtype(device)
            support_gate = has_support.to(forward_dtype)
            cond_terms.append(self.visible_image_proj(visible_mean.to(forward_dtype)) * support_gate)
            cond_terms.append(
                self.support_scale.to(forward_dtype)
                * self.support_image_proj(support_mean.to(forward_dtype))
                * support_gate
            )

        depth_visible = depth.to(torch.float32) * visible_w
        depth_mean = depth_visible.sum(dim=1) / visible_denom
        depth_var = (((depth.to(torch.float32) - depth_mean[:, None]) ** 2) * visible_w).sum(dim=1) / visible_denom
        depth_std = depth_var.clamp_min(0.0).sqrt()
        centers_b = centers.unsqueeze(0).expand(b, -1, -1)
        geom = torch.cat(
            [
                centers_b,
                visible_count[..., None] / float(v),
                support_sum[..., None] / float(v),
                support_ratio[..., None],
                torch.tanh(depth_mean)[..., None],
                torch.tanh(depth_std)[..., None],
                visible_w.amax(dim=1, keepdim=False)[..., None],
                mask_values.amax(dim=1, keepdim=False)[..., None],
            ],
            dim=-1,
        )
        forward_dtype = self._forward_dtype(device)
        cond_terms.append(
            self.geom_scale.to(forward_dtype)
            * self.geom_proj(geom.to(forward_dtype))
            * has_visible.to(forward_dtype)
        )

        cond = cond_terms[0]
        for term in cond_terms[1:]:
            cond = cond + term
        return self.out_norm(cond)
