from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def resize_intrinsics(intrinsics: torch.Tensor, old_hw: Tuple[int, int], new_hw: Tuple[int, int]) -> torch.Tensor:
    old_h, old_w = old_hw
    new_h, new_w = new_hw
    out = intrinsics.clone()
    out[..., 0, :] *= float(new_w) / float(old_w)
    out[..., 1, :] *= float(new_h) / float(old_h)
    return out


def crop_resize_with_intrinsics(
    images: torch.Tensor,
    masks: Optional[torch.Tensor],
    intrinsics: torch.Tensor,
    resolution: int = 518,
    padding_factor: float = 1.1,
    no_background: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if images.ndim != 4:
        raise ValueError(f"images should be [M,3,H,W], got {tuple(images.shape)}")
    m, c, h, w = images.shape
    if c != 3:
        raise ValueError(f"images should have 3 channels, got {c}")
    if masks is not None and masks.shape != (m, 1, h, w):
        raise ValueError(f"masks should be {(m, 1, h, w)}, got {tuple(masks.shape)}")

    device = images.device
    dtype = images.dtype
    ys_base, xs_base = torch.meshgrid(
        torch.arange(resolution, device=device, dtype=dtype),
        torch.arange(resolution, device=device, dtype=dtype),
        indexing="ij",
    )
    images_out = []
    masks_out = [] if masks is not None else None
    intrinsics_out = []

    for i in range(m):
        mask_bool = masks[i, 0] > 0.5 if masks is not None else torch.ones((h, w), device=device, dtype=torch.bool)
        if mask_bool.any():
            ys, xs = torch.where(mask_bool)
            x0 = xs.float().min()
            x1 = xs.float().max()
            y0 = ys.float().min()
            y1 = ys.float().max()
        else:
            x0 = torch.tensor(0.0, device=device)
            y0 = torch.tensor(0.0, device=device)
            x1 = torch.tensor(float(w - 1), device=device)
            y1 = torch.tensor(float(h - 1), device=device)

        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        side = torch.clamp(torch.maximum(x1 - x0 + 1.0, y1 - y0 + 1.0) * padding_factor, min=1.0)
        left = cx - side * 0.5
        top = cy - side * 0.5
        scale = float(resolution) / side

        x_in = left + (xs_base + 0.5) / scale - 0.5
        y_in = top + (ys_base + 0.5) / scale - 0.5
        grid_x = 2.0 * (x_in + 0.5) / float(w) - 1.0
        grid_y = 2.0 * (y_in + 0.5) / float(h) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        img = images[i : i + 1]
        if no_background and masks is not None:
            img = img * masks[i : i + 1]
        images_out.append(F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0])

        if masks is not None:
            masks_out.append(F.grid_sample(masks[i : i + 1], grid, mode="nearest", padding_mode="zeros", align_corners=False)[0])

        k = intrinsics[i].clone()
        k[0, 0] = k[0, 0] * scale
        k[1, 1] = k[1, 1] * scale
        k[0, 2] = (k[0, 2] - left) * scale - 0.5
        k[1, 2] = (k[1, 2] - top) * scale - 0.5
        intrinsics_out.append(k)

    return (
        torch.stack(images_out, dim=0),
        torch.stack(masks_out, dim=0) if masks_out is not None else None,
        torch.stack(intrinsics_out, dim=0),
    )


def ensure_resized_with_intrinsics(
    images: torch.Tensor,
    masks: Optional[torch.Tensor],
    intrinsics: torch.Tensor,
    resolution: int = 518,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    h, w = images.shape[-2:]
    if (h, w) == (resolution, resolution):
        return images, masks, intrinsics
    images_out = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    masks_out = F.interpolate(masks, size=(resolution, resolution), mode="nearest") if masks is not None else None
    return images_out, masks_out, resize_intrinsics(intrinsics, (h, w), (resolution, resolution))


def build_patch_ray_features(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    patch_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
    masks: Optional[torch.Tensor] = None,
    extrinsics_are_c2w: bool = True,
    camera_forward_sign: float = 1.0,
    reference_relative: bool = True,
    reference_index: int = 0,
    normalize_origins: bool = True,
) -> torch.Tensor:
    if intrinsics.ndim != 4 or extrinsics.ndim != 4:
        raise ValueError("intrinsics and extrinsics should be [B,V,...]")
    b, v = intrinsics.shape[:2]
    patch_h, patch_w = patch_hw
    image_h, image_w = image_hw
    device = intrinsics.device
    dtype = intrinsics.dtype

    c2w = extrinsics if extrinsics_are_c2w else torch.linalg.inv(extrinsics.float()).to(dtype)
    if reference_relative:
        if reference_index < 0 or reference_index >= v:
            raise ValueError(f"reference_index={reference_index} is out of range for {v} views")
        ref_c2w = c2w[:, reference_index : reference_index + 1]
        w2ref = torch.linalg.inv(ref_c2w.float()).to(dtype)
        c2w = torch.matmul(w2ref, c2w)
    r_c2w = c2w[..., :3, :3]
    centers = c2w[..., :3, 3]

    ys, xs = torch.meshgrid(
        torch.arange(patch_h, device=device, dtype=dtype),
        torch.arange(patch_w, device=device, dtype=dtype),
        indexing="ij",
    )
    u = (xs + 0.5) * (float(image_w) / float(patch_w)) - 0.5
    vv = (ys + 0.5) * (float(image_h) / float(patch_h)) - 0.5
    pix = torch.stack([u, vv, torch.ones_like(u)], dim=-1).reshape(1, 1, -1, 3)

    inv_k = torch.linalg.inv(intrinsics.float()).to(dtype)
    rays_cam = torch.matmul(inv_k.unsqueeze(2), pix.unsqueeze(-1)).squeeze(-1)
    rays_cam = F.normalize(rays_cam, dim=-1)
    rays_world = torch.matmul(r_c2w.unsqueeze(2), rays_cam.unsqueeze(-1)).squeeze(-1)
    rays_world = F.normalize(rays_world, dim=-1)

    right = F.normalize(r_c2w[..., :, 0], dim=-1)
    up = F.normalize(r_c2w[..., :, 1], dim=-1)
    forward = F.normalize(r_c2w[..., :, 2] * float(camera_forward_sign), dim=-1)

    if normalize_origins:
        if reference_relative:
            radius = centers.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-6)
            origins = centers / radius.unsqueeze(-1)
        else:
            center_mean = centers.mean(dim=1, keepdim=True)
            centered = centers - center_mean
            radius = centered.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-6)
            origins = centered / radius.unsqueeze(-1)
    else:
        origins = centers

    num_patches = patch_h * patch_w
    origins = origins.unsqueeze(2).expand(b, v, num_patches, 3)
    right = right.unsqueeze(2).expand(b, v, num_patches, 3)
    up = up.unsqueeze(2).expand(b, v, num_patches, 3)
    forward = forward.unsqueeze(2).expand(b, v, num_patches, 3)

    if masks is not None:
        mask_patch = F.interpolate(
            masks.reshape(b * v, 1, masks.shape[-2], masks.shape[-1]).float(),
            size=(patch_h, patch_w),
            mode="area",
        ).reshape(b, v, 1, num_patches).transpose(2, 3)
    else:
        mask_patch = torch.ones((b, v, num_patches, 1), device=device, dtype=dtype)

    return torch.cat([rays_world, origins, right, up, forward, mask_patch.to(dtype)], dim=-1)
