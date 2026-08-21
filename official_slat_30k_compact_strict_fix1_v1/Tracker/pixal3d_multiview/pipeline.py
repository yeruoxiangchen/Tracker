from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline

from .condition_utils import fuse_global_tokens
from .multiview_projection import (
    coords_to_centers,
    estimate_object_volume_from_visual_hull,
    resize_masks,
    resolve_object_to_world,
    sample_features_multi_view,
    scale_intrinsics_to_square,
    project_points_multi_view,
    pixal3d_grid_points,
    visual_hull_coords,
    visual_hull_front_depth_maps,
)


class Pixal3DMultiviewTo3DPipeline(Pixal3DImageTo3DPipeline):
    """Pixal3D inference wrapper with multi-view K/pose/mask projection.

    This class intentionally lives outside the Pixal3D repo. It reuses Pixal3D
    models and samplers, but replaces single-front-view projection conditioning
    with multi-view feature aggregation.
    """

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
        if not getattr(self, "_visibility_enabled", False):
            return None, {"enabled": False, "reason": "disabled"}
        if masks_sq is None or object_to_world is None:
            return None, {"enabled": False, "reason": "missing_masks_or_volume"}
        cache = getattr(self, "_front_depth_cache", None)
        if cache is None:
            cache = {}
            self._front_depth_cache = cache
        key = (
            int(image_size),
            int(getattr(self, "_vh_visibility_resolution", 48)),
            int(getattr(self, "_vh_visibility_dilation", 3)),
            float(mask_threshold),
            int(getattr(self, "_vh_min_visible_views", 1)),
            int(getattr(self, "_vh_min_support_views", 2)),
            float(getattr(self, "_vh_min_support_ratio", 0.6)),
        )
        if key not in cache:
            front_maps, stats = visual_hull_front_depth_maps(
                masks_sq,
                intrinsics_sq,
                extrinsics.to(self.device),
                extrinsics_are_c2w=extrinsics_are_c2w,
                camera_forward_sign=camera_forward_sign,
                object_to_world=object_to_world,
                resolution=int(getattr(self, "_vh_visibility_resolution", 48)),
                coordinate_size=int(image_size),
                mask_threshold=mask_threshold,
                min_visible_views=int(getattr(self, "_vh_min_visible_views", 1)),
                min_support_views=int(getattr(self, "_vh_min_support_views", 2)),
                min_support_ratio=float(getattr(self, "_vh_min_support_ratio", 0.6)),
                dilation_radius=int(getattr(self, "_vh_visibility_dilation", 3)),
            )
            cache[key] = (front_maps, stats)
        front_maps, stats = cache[key]
        return front_maps, stats

    @torch.no_grad()
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
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        device = self.device
        if self.low_vram:
            image_cond_model.to(device)

        old_grid_resolution = image_cond_model.grid_resolution
        grid_resolution = int(grid_resolution_override or old_grid_resolution)
        image_size = int(image_cond_model.image_size)
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
            visibility_depth_tolerance=float(getattr(self, "_visibility_depth_tolerance", 0.03)),
            visibility_weight_min=float(getattr(self, "_visibility_weight_min", 0.05)),
            empty_policy=empty_policy,
            fallback_weight=fallback_weight,
            support_confidence_power=support_confidence_power,
        )

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
                visibility_depth_tolerance=float(getattr(self, "_visibility_depth_tolerance", 0.03)),
                visibility_weight_min=float(getattr(self, "_visibility_weight_min", 0.05)),
                empty_policy=empty_policy,
                fallback_weight=fallback_weight,
                support_confidence_power=support_confidence_power,
            )
            z_proj = torch.cat([z_proj_lr, z_proj_hr], dim=-1)
        else:
            z_proj = z_proj_lr
            hr_stats = None

        z_global, global_stats = fuse_global_tokens(z_clstoken, z_regtokens, mode=global_fusion)
        stats = {
            "grid_resolution": grid_resolution,
            "image_size": image_size,
            "global": global_stats,
            "lr_projection": lr_stats,
            "front_depth": front_depth_stats,
        }
        if hr_stats is not None:
            stats["hr_projection"] = hr_stats

        image_cond_model.grid_resolution = old_grid_resolution
        if self.low_vram:
            image_cond_model.cpu()
        return z_global, z_proj, stats

    @torch.no_grad()
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
    ) -> dict:
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
        )
        self.last_multiview_stats["ss_condition"] = stats
        return {
            "cond": {"global": z_global, "proj": z_proj},
            "neg_cond": {"global": torch.zeros_like(z_global), "proj": torch.zeros_like(z_proj)},
        }

    @torch.no_grad()
    def get_multiview_proj_cond_shape(
        self,
        image_cond_model,
        images: list[Image.Image],
        coords: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        source_sizes: list[tuple[int, int]],
        *,
        masks: Optional[torch.Tensor] = None,
        extrinsics_are_c2w: bool = True,
        camera_forward_sign: float = 1.0,
        object_to_world: Optional[torch.Tensor] = None,
        grid_resolution_override: Optional[int] = None,
        mask_threshold: float = 0.5,
        empty_policy: str = "zero",
        fallback_weight: float = 1.0,
        support_confidence_power: float = 1.0,
        global_fusion: str = "concat",
        stats_key: str = "shape_condition",
    ) -> dict:
        grid_resolution = int(grid_resolution_override or image_cond_model.grid_resolution)
        z_global, z_proj_dense, stats = self._extract_multiview_proj(
            image_cond_model,
            images,
            intrinsics,
            extrinsics,
            source_sizes,
            masks=masks,
            extrinsics_are_c2w=extrinsics_are_c2w,
            camera_forward_sign=camera_forward_sign,
            object_to_world=object_to_world,
            grid_resolution_override=grid_resolution,
            mask_threshold=mask_threshold,
            empty_policy=empty_policy,
            fallback_weight=fallback_weight,
            support_confidence_power=support_confidence_power,
            global_fusion=global_fusion,
        )
        coords = coords.int()
        dense = z_proj_dense.reshape(1, grid_resolution, grid_resolution, grid_resolution, -1)
        batch = coords[:, 0].long()
        x = coords[:, 1].long()
        y = coords[:, 2].long()
        z = coords[:, 3].long()
        if (x.min() < 0 or y.min() < 0 or z.min() < 0 or x.max() >= grid_resolution or y.max() >= grid_resolution or z.max() >= grid_resolution):
            raise ValueError(f"coords out of projection grid range {grid_resolution}: min={coords[:, 1:].min().item()} max={coords[:, 1:].max().item()}")
        z_proj_sparse = dense[batch, x, y, z]
        proj_sparse = SparseTensor(feats=z_proj_sparse, coords=coords)
        self.last_multiview_stats[stats_key] = stats
        return {
            "cond": {"global": z_global, "proj": proj_sparse},
            "neg_cond": {"global": torch.zeros_like(z_global), "proj": SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords)},
        }

    def _visual_hull_sparse_coords(
        self,
        masks: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        source_sizes: list[tuple[int, int]],
        *,
        extrinsics_are_c2w: bool,
        camera_forward_sign: float,
        object_to_world: Optional[torch.Tensor],
        mask_threshold: float,
        min_visible_views: int,
        min_support_views: int,
        min_support_ratio: float,
        surface_only: bool,
        max_coords: int,
        seed: int,
    ) -> torch.Tensor:
        image_size = int(self.image_cond_model_ss.image_size)
        intrinsics_sq = scale_intrinsics_to_square(intrinsics, source_sizes, image_size, self.device)
        masks_sq = resize_masks(masks, image_size, self.device)
        coords, stats = visual_hull_coords(
            masks_sq,
            intrinsics_sq,
            extrinsics.to(self.device),
            extrinsics_are_c2w=extrinsics_are_c2w,
            camera_forward_sign=camera_forward_sign,
            object_to_world=object_to_world,
            resolution=32,
            mask_threshold=mask_threshold,
            min_visible_views=min_visible_views,
            min_support_views=min_support_views,
            min_support_ratio=min_support_ratio,
            surface_only=surface_only,
        )
        if max_coords > 0 and coords.shape[0] > max_coords:
            generator = torch.Generator(device=coords.device)
            generator.manual_seed(int(seed))
            keep = torch.randperm(coords.shape[0], generator=generator, device=coords.device)[:max_coords]
            coords = coords[keep]
        if coords.shape[0] == 0:
            raise ValueError("visual hull produced 0 coords; relax mask/support thresholds or check camera poses")
        batch = torch.zeros((coords.shape[0], 1), device=coords.device, dtype=torch.int32)
        self.last_multiview_stats["visual_hull_coords"] = stats.to_dict()
        self.last_multiview_stats["visual_hull_coords"]["num_coords_after_limit"] = int(coords.shape[0])
        return torch.cat([batch, coords.to(torch.int32)], dim=1)

    @torch.no_grad()
    def run_multiview(
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
        world_to_object: Optional[torch.Tensor] = None,
        coords_source: str = "network",
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        mask_threshold: float = 0.5,
        vh_min_visible_views: int = 1,
        vh_min_support_views: int = 2,
        vh_min_support_ratio: float = 0.6,
        vh_surface_only: bool = True,
        vh_max_coords: int = 12000,
        vh_volume_resolution: int = 48,
        vh_volume_initial_extent_ratio: float = 0.6,
        vh_volume_padding: float = 1.25,
        vh_volume_min_extent: float = 0.05,
        vh_volume_refine_steps: int = 2,
        visibility_enabled: bool = True,
        vh_visibility_resolution: int = 48,
        vh_visibility_dilation: int = 3,
        visibility_depth_tolerance: float = 0.0,
        visibility_depth_tolerance_ratio: float = 0.15,
        visibility_weight_min: float = 0.05,
        empty_policy: str = "zero",
        fallback_weight: float = 1.0,
        support_confidence_power: float = 1.0,
        global_fusion: str = "concat",
    ):
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == "1024_cascade":
            hr_resolution = 1024
        elif pipeline_type == "1536_cascade":
            hr_resolution = 1536
        else:
            raise ValueError(f"Unsupported pipeline_type={pipeline_type}")
        assert self.image_cond_model_ss is not None, "image_cond_model_ss not set"
        assert self.image_cond_model_shape_512 is not None, "image_cond_model_shape_512 not set"
        assert self.image_cond_model_shape_1024 is not None, "image_cond_model_shape_1024 not set"
        assert self.image_cond_model_tex_1024 is not None, "image_cond_model_tex_1024 not set"
        if int(num_samples) != 1:
            raise NotImplementedError(
                "Pixal3DMultiviewTo3DPipeline currently supports num_samples=1 only. "
                "The multi-view projection condition is built for one shared view set, "
                "while downstream sparse coords may contain multiple batch ids."
            )
        if coords_source == "visual_hull" and masks is None:
            raise ValueError("coords_source=visual_hull requires masks")

        intrinsics = intrinsics.to(self.device).float()
        extrinsics = extrinsics.to(self.device).float()
        if masks is not None:
            masks = masks.to(self.device).float()

        object_to_world_override = resolve_object_to_world(object_to_world, world_to_object, self.device)
        volume_stats = None
        volume_extent = None
        if object_to_world_override is not None:
            object_to_world = object_to_world_override
            volume_mode = "debug_override"
            volume_stats = {"mode": volume_mode, "object_to_world": object_to_world.detach().cpu().tolist()}
            volume_extent = float(torch.linalg.norm(object_to_world[:3, :3], dim=0).max().item())
        elif masks is not None:
            estimate = estimate_object_volume_from_visual_hull(
                masks,
                intrinsics,
                extrinsics,
                extrinsics_are_c2w=extrinsics_are_c2w,
                camera_forward_sign=camera_forward_sign,
                mask_threshold=mask_threshold,
                resolution=vh_volume_resolution,
                min_visible_views=vh_min_visible_views,
                min_support_views=vh_min_support_views,
                min_support_ratio=vh_min_support_ratio,
                initial_extent_ratio=vh_volume_initial_extent_ratio,
                padding=vh_volume_padding,
                min_extent=vh_volume_min_extent,
                refine_steps=vh_volume_refine_steps,
            )
            object_to_world = estimate.object_to_world
            volume_mode = "visual_hull_auto"
            volume_stats = estimate.to_dict()
            volume_extent = float(estimate.extent_world)
        else:
            object_to_world = None
            volume_mode = "none"
            volume_stats = {"mode": volume_mode}
            volume_extent = None

        if visibility_depth_tolerance > 0:
            depth_tolerance = float(visibility_depth_tolerance)
        elif volume_extent is not None:
            depth_tolerance = max(float(volume_extent) * float(visibility_depth_tolerance_ratio), 1e-4)
        else:
            depth_tolerance = 0.03
        self._front_depth_cache = {}
        self._visibility_enabled = bool(visibility_enabled and masks is not None and object_to_world is not None)
        self._visibility_depth_tolerance = depth_tolerance
        self._visibility_weight_min = float(visibility_weight_min)
        self._vh_visibility_resolution = int(vh_visibility_resolution)
        self._vh_visibility_dilation = int(vh_visibility_dilation)
        self._vh_min_visible_views = int(vh_min_visible_views)
        self._vh_min_support_views = int(vh_min_support_views)
        self._vh_min_support_ratio = float(vh_min_support_ratio)

        self.last_multiview_stats = {
            "num_views": len(images),
            "coords_source": coords_source,
            "extrinsics_are_c2w": bool(extrinsics_are_c2w),
            "camera_forward_sign": float(camera_forward_sign),
            "object_volume_mode": volume_mode,
            "object_volume_estimate": volume_stats,
            "mesh_output_space": "pixal3d_canonical_object_space",
            "note": "object_volume_estimate is only used for internal projection sampling; output mesh is not transformed to world.",
            "projection_fallback": {
                "empty_policy": empty_policy,
                "fallback_weight": float(fallback_weight),
                "support_confidence_power": float(support_confidence_power),
                "global_fusion": global_fusion,
            },
            "visibility": {
                "enabled": bool(self._visibility_enabled),
                "depth_tolerance": float(depth_tolerance),
                "depth_tolerance_ratio": float(visibility_depth_tolerance_ratio),
                "weight_min": float(visibility_weight_min),
                "front_depth_resolution": int(vh_visibility_resolution),
                "front_depth_dilation": int(vh_visibility_dilation),
            },
        }

        torch.manual_seed(seed)
        if coords_source == "visual_hull":
            coords = self._visual_hull_sparse_coords(
                masks,
                intrinsics,
                extrinsics,
                source_sizes,
                extrinsics_are_c2w=extrinsics_are_c2w,
                camera_forward_sign=camera_forward_sign,
                object_to_world=object_to_world,
                mask_threshold=mask_threshold,
                min_visible_views=vh_min_visible_views,
                min_support_views=vh_min_support_views,
                min_support_ratio=vh_min_support_ratio,
                surface_only=vh_surface_only,
                max_coords=vh_max_coords,
                seed=seed,
            )
        elif coords_source == "network":
            cond_ss = self.get_multiview_proj_cond_ss(
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
            )
            coords = self.sample_sparse_structure(cond_ss, 32, num_samples, sparse_structure_sampler_params)
            del cond_ss
            torch.cuda.empty_cache()
        else:
            raise ValueError(f"Unknown coords_source={coords_source}")
        self.last_multiview_stats["num_sparse_coords"] = int(coords.shape[0])

        cond_shape_lr = self.get_multiview_proj_cond_shape(
            self.image_cond_model_shape_512,
            images,
            coords,
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
            stats_key="shape_lr_condition",
        )
        lr_slat = self.sample_shape_slat(cond_shape_lr, self.models["shape_slat_flow_model_512"], coords, shape_slat_sampler_params)
        del cond_shape_lr
        torch.cuda.empty_cache()

        if self.low_vram:
            self.models["shape_slat_decoder"].to(self.device)
            self.models["shape_slat_decoder"].low_vram = True
        hr_coords = self.models["shape_slat_decoder"].upsample(lr_slat, upsample_times=4)
        if self.low_vram:
            self.models["shape_slat_decoder"].cpu()
            self.models["shape_slat_decoder"].low_vram = False

        lr_resolution = 512
        actual_hr_resolution = hr_resolution
        while True:
            grid_res = actual_hr_resolution // 16
            quant_coords = torch.cat(
                [
                    hr_coords[:, :1],
                    ((hr_coords[:, 1:] + 0.5) / lr_resolution * (grid_res - 1)).round().int(),
                ],
                dim=1,
            )
            hr_coords_unique = quant_coords.unique(dim=0)
            if hr_coords_unique.shape[0] < max_num_tokens or actual_hr_resolution == 1024:
                break
            actual_hr_resolution -= 128
        actual_grid_res = actual_hr_resolution // 16
        del lr_slat, hr_coords, quant_coords
        torch.cuda.empty_cache()

        cond_shape_hr = self.get_multiview_proj_cond_shape(
            self.image_cond_model_shape_1024,
            images,
            hr_coords_unique,
            intrinsics,
            extrinsics,
            source_sizes,
            masks=masks,
            extrinsics_are_c2w=extrinsics_are_c2w,
            camera_forward_sign=camera_forward_sign,
            object_to_world=object_to_world,
            grid_resolution_override=actual_grid_res,
            mask_threshold=mask_threshold,
            empty_policy=empty_policy,
            fallback_weight=fallback_weight,
            support_confidence_power=support_confidence_power,
            global_fusion=global_fusion,
            stats_key="shape_hr_condition",
        )
        noise_hr = SparseTensor(
            feats=torch.randn(hr_coords_unique.shape[0], self.models["shape_slat_flow_model_1024"].in_channels).to(self.device),
            coords=hr_coords_unique,
        )
        sampler_params_hr = {**self.shape_slat_sampler_params, **shape_slat_sampler_params}
        flow_model_hr = self.models["shape_slat_flow_model_1024"]
        if self.low_vram:
            flow_model_hr.to(self.device)
        hr_slat = self.shape_slat_sampler.sample(
            flow_model_hr,
            noise_hr,
            **cond_shape_hr,
            **sampler_params_hr,
            verbose=True,
            tqdm_desc=f"Sampling HR shape SLat (multiview proj, {actual_hr_resolution})",
        ).samples
        if self.low_vram:
            flow_model_hr.cpu()
        std = torch.tensor(self.shape_slat_normalization["std"])[None].to(hr_slat.device)
        mean = torch.tensor(self.shape_slat_normalization["mean"])[None].to(hr_slat.device)
        shape_slat = hr_slat * std + mean
        del cond_shape_hr, noise_hr, hr_slat, hr_coords_unique
        torch.cuda.empty_cache()

        cond_tex = self.get_multiview_proj_cond_shape(
            self.image_cond_model_tex_1024,
            images,
            shape_slat.coords,
            intrinsics,
            extrinsics,
            source_sizes,
            masks=masks,
            extrinsics_are_c2w=extrinsics_are_c2w,
            camera_forward_sign=camera_forward_sign,
            object_to_world=object_to_world,
            grid_resolution_override=actual_grid_res,
            mask_threshold=mask_threshold,
            empty_policy=empty_policy,
            fallback_weight=fallback_weight,
            support_confidence_power=support_confidence_power,
            global_fusion=global_fusion,
            stats_key="texture_condition",
        )
        tex_slat = self.sample_tex_slat(cond_tex, self.models["tex_slat_flow_model_1024"], shape_slat, tex_slat_sampler_params)
        del cond_tex
        torch.cuda.empty_cache()

        out_mesh = self.decode_latent(shape_slat, tex_slat, actual_hr_resolution)
        self.last_multiview_stats["actual_hr_resolution"] = int(actual_hr_resolution)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, actual_hr_resolution)
        return out_mesh
