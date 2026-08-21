from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from .condition_utils import fuse_global_tokens
from .multiview_projection import (
    geometry_features_from_projection,
    pixal3d_grid_points,
    project_points_multi_view,
    resize_masks,
    sample_features_multi_view,
    scale_intrinsics_to_square,
    visual_hull_front_depth_maps,
)
from .view_aggregator import sample_view_features_for_aggregation


class SparseMultiviewConditionBuilder:
    """Lightweight multi-view condition builder for sparse-flow training.

    This avoids importing Pixal3D's full mesh pipeline, which requires optional
    mesh runtime dependencies such as cumesh. Sparse training only needs image
    features projected onto the sparse-structure grid.
    """

    def __init__(self, device: torch.device, low_vram: bool = False):
        self._device = device
        self.low_vram = bool(low_vram)
        self.last_multiview_stats: dict = {}
        self._front_depth_cache: dict = {}
        self._visibility_enabled = False
        self._visibility_depth_tolerance = 0.03
        self._visibility_weight_min = 0.05
        self._vh_visibility_resolution = 48
        self._vh_visibility_dilation = 3
        self._vh_min_visible_views = 1
        self._vh_min_support_views = 2
        self._vh_min_support_ratio = 0.6
        self.image_cond_model_ss = None
        self.view_aggregator = None
        self.geometry_adapter = None
        self.pose_consistency_head = None
        self.pose_consistency_alpha = 1.0
        self.last_pose_consistency_tensors = None

    @property
    def device(self) -> torch.device:
        return self._device

    def _images_to_tensor(self, image_cond_model, images: list[Image.Image]) -> torch.Tensor:
        tensors = []
        for image in images:
            image = image.resize((image_cond_model.image_size, image_cond_model.image_size), Image.LANCZOS)
            arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float())
        return torch.stack(tensors, dim=0).to(self.device)

    def _get_front_depth_maps(
        self,
        masks_sq: Optional[torch.Tensor],
        intrinsics_sq: torch.Tensor,
        extrinsics: torch.Tensor,
        image_size: int,
        *,
        extrinsics_are_c2w: bool,
        camera_forward_sign: float,
        object_to_world: Optional[torch.Tensor],
        mask_threshold: float,
    ) -> tuple[Optional[torch.Tensor], dict]:
        if not self._visibility_enabled:
            return None, {"enabled": False, "reason": "disabled"}
        if masks_sq is None or object_to_world is None:
            return None, {"enabled": False, "reason": "missing_masks_or_volume"}

        key = (
            int(image_size),
            int(self._vh_visibility_resolution),
            int(self._vh_visibility_dilation),
            float(mask_threshold),
            int(self._vh_min_visible_views),
            int(self._vh_min_support_views),
            float(self._vh_min_support_ratio),
        )
        if key not in self._front_depth_cache:
            front_maps, stats = visual_hull_front_depth_maps(
                masks_sq,
                intrinsics_sq,
                extrinsics.to(self.device),
                extrinsics_are_c2w=extrinsics_are_c2w,
                camera_forward_sign=camera_forward_sign,
                object_to_world=object_to_world,
                resolution=int(self._vh_visibility_resolution),
                coordinate_size=int(image_size),
                mask_threshold=mask_threshold,
                min_visible_views=int(self._vh_min_visible_views),
                min_support_views=int(self._vh_min_support_views),
                min_support_ratio=float(self._vh_min_support_ratio),
                dilation_radius=int(self._vh_visibility_dilation),
            )
            self._front_depth_cache[key] = (front_maps, stats)
        return self._front_depth_cache[key]

    def _extract_multiview_proj(
        self,
        image_cond_model,
        images: list[Image.Image],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        source_sizes: list[tuple[int, int]],
        *,
        masks: Optional[torch.Tensor],
        extrinsics_are_c2w: bool,
        camera_forward_sign: float,
        object_to_world: Optional[torch.Tensor],
        grid_resolution_override: Optional[int] = None,
        mask_threshold: float = 0.5,
        empty_policy: str = "zero",
        fallback_weight: float = 1.0,
        support_confidence_power: float = 1.0,
        global_fusion: str = "concat",
        geometry_feature_mode: str = "none",
        geometry_feature_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        device = self.device
        if self.low_vram:
            image_cond_model.to(device)

        old_grid_resolution = image_cond_model.grid_resolution
        grid_resolution = int(grid_resolution_override or old_grid_resolution)
        image_size = int(image_cond_model.image_size)
        with torch.no_grad():
            intrinsics_sq = scale_intrinsics_to_square(intrinsics, source_sizes, image_size, device)
            masks_sq = resize_masks(masks, image_size, device)
            images_tensor = self._images_to_tensor(image_cond_model, images)

            points_obj, _ = pixal3d_grid_points(grid_resolution, device=device, dtype=torch.float32)
            points_2d, depths, valid_depth = project_points_multi_view(
                points_obj,
                intrinsics_sq,
                extrinsics.to(device),
                extrinsics_are_c2w=extrinsics_are_c2w,
                camera_forward_sign=camera_forward_sign,
                object_to_world=object_to_world,
            )
            front_depth_maps, front_depth_stats = self._get_front_depth_maps(
                masks_sq,
                intrinsics_sq,
                extrinsics,
                image_size,
                extrinsics_are_c2w=extrinsics_are_c2w,
                camera_forward_sign=camera_forward_sign,
                object_to_world=object_to_world,
                mask_threshold=mask_threshold,
            )

            if image_cond_model.use_naf_upsample:
                image_for_naf = images_tensor.clone()
            dino_input = image_cond_model.transform(images_tensor)
            z = image_cond_model.extract_features(dino_input)

            z_clstoken = z[:, 0:1]
            num_reg = getattr(image_cond_model.model.config, "num_register_tokens", 4)
            z_regtokens = z[:, 1 : 1 + num_reg]
            z_patchtokens = z[:, 1 + num_reg :]
            z_patchtokens_spatial = z_patchtokens.reshape(
                images_tensor.shape[0],
                image_cond_model.patch_number,
                image_cond_model.patch_number,
                -1,
            )

            z_proj_lr, lr_stats = sample_features_multi_view(
                z_patchtokens_spatial,
                points_2d,
                valid_depth,
                coordinate_size=image_size,
                masks=masks_sq,
                mask_threshold=mask_threshold,
                depths=depths,
                front_depth_maps=front_depth_maps,
                visibility_depth_tolerance=float(self._visibility_depth_tolerance),
                visibility_weight_min=float(self._visibility_weight_min),
                empty_policy=empty_policy,
                fallback_weight=fallback_weight,
                support_confidence_power=support_confidence_power,
            )
            view_tokens = None
            view_token_stats = None
            if self.view_aggregator is not None or self.pose_consistency_head is not None:
                if image_cond_model.use_naf_upsample:
                    raise ValueError("view/pose aggregation currently supports the LR DINO projection path only")
                sampled_views, support_weights, view_geom, view_token_stats = sample_view_features_for_aggregation(
                    z_patchtokens_spatial,
                    points_obj,
                    points_2d,
                    depths,
                    valid_depth,
                    coordinate_size=image_size,
                    masks=masks_sq,
                    mask_threshold=mask_threshold,
                    front_depth_maps=front_depth_maps,
                    visibility_depth_tolerance=float(self._visibility_depth_tolerance),
                    visibility_weight_min=float(self._visibility_weight_min),
                )
                view_tokens = (sampled_views.detach(), support_weights.detach(), view_geom.detach())

            if image_cond_model.use_naf_upsample:
                image_cond_model._load_naf()
                lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2)
                hr_features = image_cond_model.naf_model(image_for_naf, lr_features_bchw, image_cond_model.naf_target_size)
                z_proj_hr, hr_stats = sample_features_multi_view(
                    hr_features,
                    points_2d,
                    valid_depth,
                    coordinate_size=image_size,
                    masks=masks_sq,
                    mask_threshold=mask_threshold,
                    depths=depths,
                    front_depth_maps=front_depth_maps,
                    visibility_depth_tolerance=float(self._visibility_depth_tolerance),
                    visibility_weight_min=float(self._visibility_weight_min),
                    empty_policy=empty_policy,
                    fallback_weight=fallback_weight,
                    support_confidence_power=support_confidence_power,
                )
                z_proj = torch.cat([z_proj_lr, z_proj_hr], dim=-1)
            else:
                z_proj = z_proj_lr
                hr_stats = None

            z_global, global_stats = fuse_global_tokens(z_clstoken, z_regtokens, mode=global_fusion)

        pose_consistency_stats = {"enabled": False}
        self.last_pose_consistency_tensors = None
        if self.pose_consistency_head is not None:
            if view_tokens is None:
                raise RuntimeError("pose_consistency_head requested but view tokens were not computed")
            sampled_views, support_weights, view_geom = view_tokens
            _, pose_consistency_stats, pose_tensors = self.pose_consistency_head(
                sampled_views.detach(),
                support_weights.detach(),
                view_geom.detach(),
            )
            self.last_pose_consistency_tensors = pose_tensors

        aggregator_stats = {"enabled": False}
        if self.view_aggregator is not None:
            sampled_views, support_weights, view_geom = view_tokens
            consistency_logits = None
            if self.last_pose_consistency_tensors is not None:
                consistency_logits = self.last_pose_consistency_tensors.get("logits")
            z_proj, aggregator_stats = self.view_aggregator(
                z_proj.detach(),
                sampled_views,
                support_weights,
                view_geom,
                consistency_logits=consistency_logits,
                consistency_alpha=float(self.pose_consistency_alpha),
            )
            aggregator_stats["view_tokens"] = view_token_stats

        geometry_stats = {"enabled": False, "mode": geometry_feature_mode}
        geometry_adapter_stats = {"enabled": False}
        geometry_features = None
        if geometry_feature_mode != "none" or self.geometry_adapter is not None:
            with torch.no_grad():
                geometry_features, geometry_stats = geometry_features_from_projection(
                    points_obj,
                    points_2d,
                    depths,
                    valid_depth,
                    coordinate_size=image_size,
                    masks=masks_sq,
                    mask_threshold=mask_threshold,
                    front_depth_maps=front_depth_maps,
                    visibility_depth_tolerance=float(self._visibility_depth_tolerance),
                    visibility_weight_min=float(self._visibility_weight_min),
                    min_visible_views=int(self._vh_min_visible_views),
                    min_support_views=int(self._vh_min_support_views),
                    min_support_ratio=float(self._vh_min_support_ratio),
                )
            geometry_features = geometry_features.to(device=z_proj.device, dtype=z_proj.dtype).unsqueeze(0)

        if self.geometry_adapter is not None:
            if geometry_features is None:
                raise RuntimeError("geometry_adapter requested but geometry_features were not computed")
            adapter_features = geometry_features.squeeze(0).to(dtype=torch.float32)
            z_proj, geometry_adapter_stats = self.geometry_adapter(z_proj, adapter_features)

        if geometry_feature_mode != "none":
            if geometry_features is None:
                raise RuntimeError("geometry_feature_mode requested but geometry_features were not computed")
            channels = int(min(z_proj.shape[-1], geometry_features.shape[-1]))
            z_proj = z_proj.clone()
            if geometry_feature_mode == "add":
                z_proj[..., :channels] = z_proj[..., :channels] + float(geometry_feature_scale) * geometry_features[..., :channels]
            elif geometry_feature_mode == "replace":
                z_proj[..., :channels] = float(geometry_feature_scale) * geometry_features[..., :channels]
            else:
                raise ValueError(f"Unknown geometry_feature_mode: {geometry_feature_mode}")
            geometry_stats.update(
                {
                    "mode": geometry_feature_mode,
                    "scale": float(geometry_feature_scale),
                    "injected_channels": channels,
                }
            )

        stats = {
            "grid_resolution": grid_resolution,
            "image_size": image_size,
            "global": global_stats,
            "lr_projection": lr_stats,
            "front_depth": front_depth_stats,
            "geometry_features": geometry_stats,
            "geometry_adapter": geometry_adapter_stats,
            "view_aggregator": aggregator_stats,
            "pose_consistency": pose_consistency_stats,
        }
        if hr_stats is not None:
            stats["hr_projection"] = hr_stats

        image_cond_model.grid_resolution = old_grid_resolution
        if self.low_vram:
            image_cond_model.cpu()
        return z_global, z_proj, stats

    def get_multiview_proj_cond_ss(
        self,
        images: list[Image.Image],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        source_sizes: list[tuple[int, int]],
        *,
        masks: Optional[torch.Tensor] = None,
        extrinsics_are_c2w: bool = True,
        camera_forward_sign: float = 1.0,
        object_to_world: Optional[torch.Tensor] = None,
        mask_threshold: float = 0.5,
        empty_policy: str = "zero",
        fallback_weight: float = 1.0,
        support_confidence_power: float = 1.0,
        global_fusion: str = "concat",
        geometry_feature_mode: str = "none",
        geometry_feature_scale: float = 1.0,
    ) -> dict:
        self._front_depth_cache = {}
        z_global, z_proj, stats = self._extract_multiview_proj(
            self.image_cond_model_ss,
            images,
            intrinsics,
            extrinsics,
            source_sizes,
            masks=masks,
            extrinsics_are_c2w=extrinsics_are_c2w,
            camera_forward_sign=camera_forward_sign,
            object_to_world=object_to_world,
            mask_threshold=mask_threshold,
            empty_policy=empty_policy,
            fallback_weight=fallback_weight,
            support_confidence_power=support_confidence_power,
            global_fusion=global_fusion,
            geometry_feature_mode=geometry_feature_mode,
            geometry_feature_scale=geometry_feature_scale,
        )
        self.last_multiview_stats["ss_condition"] = stats
        return {
            "cond": {"global": z_global, "proj": z_proj},
            "neg_cond": {"global": torch.zeros_like(z_global), "proj": torch.zeros_like(z_proj)},
        }
