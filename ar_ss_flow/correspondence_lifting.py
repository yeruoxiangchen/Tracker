from __future__ import annotations

import sys
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import torch
import torch.nn.functional as F
from torch import nn

# Support both `python -m ar_ss_flow.<module>` and direct script execution.
TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from ar_ss_flow.pose_lifting import (
    LIFTING_METADATA_NAMES,
    SUPPORT_METADATA_INDEX,
    _prior_volume_features,
    _sample_maps,
    build_projection_geometry,
)


CORRESPONDENCE_MODEL_VERSION = "ar_ss_flow.local_voxel_correspondence.pairwise_v3"
CORRESPONDENCE_VOLUME_VERSION = "ar_ss_flow.correspondence_lifting_volume.pairwise_v3"

POSE_NEGATIVE_MODES = (
    "pose_cyclic1",
    "pose_cyclic2",
    "pose_reverse",
)
CORRESPONDENCE_NEGATIVE_MODES = (*POSE_NEGATIVE_MODES, "cross_sample")

PER_VIEW_GEOMETRY_NAMES = (
    "valid",
    "mask_weight",
    "depth_confidence_weight",
    "depth_consistency_weight",
    "depth_residual_normalized",
    "combined_weight",
    "ray_x",
    "ray_y",
    "ray_z",
    "camera_depth_normalized",
)

CORRESPONDENCE_METADATA_NAMES = (
    *LIFTING_METADATA_NAMES,
    "correspondence_confidence",
    "correspondence_disagreement",
    "effective_view_fraction",
)
CORRESPONDENCE_CONFIDENCE_INDEX = CORRESPONDENCE_METADATA_NAMES.index(
    "correspondence_confidence"
)
CORRESPONDENCE_DISAGREEMENT_INDEX = CORRESPONDENCE_METADATA_NAMES.index(
    "correspondence_disagreement"
)
EFFECTIVE_VIEW_FRACTION_INDEX = CORRESPONDENCE_METADATA_NAMES.index(
    "effective_view_fraction"
)

PAIR_FEATURE_EXPORT_VERSION = "ar_ss_flow.local_pair_features.v1"
PAIR_FEATURE_SCALAR_NAMES = (
    "pairwise_logit",
    "pairwise_probability",
    "minimum_physical_weight",
    "mean_physical_weight",
    "absolute_depth_residual_difference",
    "ray_cosine",
    "absolute_camera_depth_difference",
    "geometry_similarity",
)


def pair_feature_dim(pairwise_dim: int) -> int:
    return 3 * int(pairwise_dim) + len(PAIR_FEATURE_SCALAR_NAMES)


def correspondence_schema_hash() -> str:
    text = "\n".join(("pairwise_before_aggregation_v3", *PER_VIEW_GEOMETRY_NAMES, *CORRESPONDENCE_METADATA_NAMES))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_csv(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("CSV value must be non-empty")
    return values


def pose_variant_extrinsics(
    extrinsics: torch.Tensor,
    mode: Literal[
        "correct",
        "pose_cyclic1",
        "pose_cyclic2",
        "pose_reverse",
    ],
    *,
    heldout_index: int | None = None,
) -> torch.Tensor:
    """Return a deterministic image-pose binding corruption.

    When ``heldout_index`` is provided, only source-view camera matrices are
    permuted.  The held-out camera remains bit-exact, so correct and wrong
    branches are always scored against the same held-out observation.
    """

    if mode == "correct":
        return extrinsics
    views = int(extrinsics.shape[0])
    if views < 2:
        return extrinsics

    if heldout_index is not None:
        if not 0 <= int(heldout_index) < views:
            raise IndexError(f"heldout view {heldout_index} outside {views}")
        source_ids = [index for index in range(views) if index != int(heldout_index)]
        if len(source_ids) < 2:
            return extrinsics
        source_index = torch.as_tensor(source_ids, device=extrinsics.device, dtype=torch.long)
        source = extrinsics.index_select(0, source_index)
        corrupted_source = pose_variant_extrinsics(source, mode, heldout_index=None)
        result = extrinsics.clone()
        result.index_copy_(0, source_index, corrupted_source)
        return result

    if mode == "pose_cyclic1":
        return torch.roll(extrinsics, shifts=1, dims=0)
    if mode == "pose_cyclic2":
        return torch.roll(extrinsics, shifts=max(1, min(2, views - 1)), dims=0)
    if mode == "pose_reverse":
        return torch.flip(extrinsics, dims=(0,))
    raise ValueError(f"unsupported correspondence pose mode={mode!r}")


def deterministic_view_subset(view_count: int, requested: int) -> list[int]:
    if view_count <= 0:
        raise ValueError("view_count must be positive")
    if requested <= 0 or requested >= view_count:
        return list(range(view_count))
    if requested == 1:
        return [0]
    positions = torch.linspace(0, view_count - 1, requested).round().long().tolist()
    result: list[int] = []
    for index in positions:
        if index not in result:
            result.append(int(index))
    cursor = 0
    while len(result) < requested:
        if cursor not in result:
            result.append(cursor)
        cursor += 1
    return sorted(result[:requested])


def subset_sample_views(sample: dict[str, Any], view_indices: Sequence[int]) -> dict[str, Any]:
    indices = torch.as_tensor(view_indices, dtype=torch.long)
    result = dict(sample)
    for key in (
        "visual_patch_features",
        "predicted_depth",
        "depth_confidence",
        "masks",
        "intrinsics",
        "extrinsics",
    ):
        value = sample[key]
        result[key] = value.index_select(0, indices)
    # Cached projection geometry is tied to the full view layout.
    result.pop("correct_geometry", None)
    return result


def _depth_maps_for_mode(
    predicted_depth: torch.Tensor,
    mode: str,
    *,
    depth_corruption_scale: float,
) -> torch.Tensor:
    depth_maps = predicted_depth[:, None].float()
    if mode != "depth_corrupt":
        return depth_maps
    height, width = predicted_depth.shape[-2:]
    depth_maps = torch.flip(
        torch.roll(
            depth_maps,
            shifts=(max(1, height // 6), max(1, width // 5)),
            dims=(-2, -1),
        ),
        dims=(-1,),
    )
    view_scale = torch.linspace(
        0.80,
        1.20,
        int(depth_maps.shape[0]),
        device=depth_maps.device,
        dtype=depth_maps.dtype,
    ).reshape(-1, 1, 1, 1)
    return depth_maps * view_scale * float(depth_corruption_scale)


def sample_per_view_voxel_evidence(
    *,
    visual_patch_features: torch.Tensor,
    predicted_depth: torch.Tensor,
    depth_confidence: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    calibration: dict[str, Any],
    grid_transform: str,
    extrinsics_type: str,
    camera_forward_sign: float,
    mode: str = "correct",
    volume_side: int = 16,
    depth_corruption_scale: float = 1.15,
    object_to_world: torch.Tensor | None = None,
    heldout_index: int | None = None,
) -> dict[str, torch.Tensor | bool | str | int | float]:
    """Sample raw per-view evidence at every 16^3 voxel without aggregating views."""

    if visual_patch_features.ndim != 3:
        raise ValueError(
            "visual_patch_features must be [V,P,C], got "
            f"{tuple(visual_patch_features.shape)}"
        )
    views, patch_count, channels = map(int, visual_patch_features.shape)
    patch_side = int(round(math.sqrt(patch_count)))
    if patch_side * patch_side != patch_count:
        raise ValueError(f"patch count is not square: {patch_count}")
    if predicted_depth.ndim != 3 or depth_confidence.shape != predicted_depth.shape:
        raise ValueError("depth/confidence must be aligned [V,H,W]")
    if masks.shape != predicted_depth.shape or int(masks.shape[0]) != views:
        raise ValueError("mask/depth/visual view count mismatch")
    if intrinsics.shape != (views, 3, 3) or extrinsics.shape != (views, 4, 4):
        raise ValueError("K/T view count mismatch")

    device = visual_patch_features.device
    image_height, image_width = map(int, predicted_depth.shape[-2:])
    geometry_mode = "correct" if mode == "depth_corrupt" else mode
    if geometry_mode not in ("correct", *POSE_NEGATIVE_MODES):
        raise ValueError(f"unsupported evidence mode={mode!r}")
    variant_extrinsics = pose_variant_extrinsics(
        extrinsics, geometry_mode, heldout_index=heldout_index
    )
    geometry = build_projection_geometry(
        intrinsics=intrinsics,
        extrinsics=variant_extrinsics,
        grid_transform=grid_transform,
        extrinsics_type=extrinsics_type,
        camera_forward_sign=camera_forward_sign,
        image_height=image_height,
        image_width=image_width,
        patch_grid_side=patch_side,
        volume_side=volume_side,
        object_to_world=object_to_world,
    )

    patch_maps = visual_patch_features.permute(0, 2, 1).reshape(
        views, channels, patch_side, patch_side
    )
    sampled_visual = _sample_maps(patch_maps.float(), geometry["patch_grid"].float())
    depth_maps = _depth_maps_for_mode(
        predicted_depth,
        mode,
        depth_corruption_scale=depth_corruption_scale,
    )
    sampled_depth = _sample_maps(depth_maps, geometry["image_grid"].float())[:, 0]
    sampled_confidence = _sample_maps(
        depth_confidence[:, None].float(), geometry["image_grid"].float()
    )[:, 0]
    sampled_mask = _sample_maps(
        masks[:, None].float(), geometry["image_grid"].float()
    )[:, 0].clamp(0.0, 1.0)
    valid = geometry["valid"].float()
    mask_weight = sampled_mask * valid
    positive_conf = sampled_confidence[sampled_confidence > 0]
    conf_scale = (
        positive_conf.median().clamp_min(1.0e-6)
        if positive_conf.numel()
        else sampled_confidence.new_tensor(1.0)
    )
    confidence_weight = (sampled_confidence / conf_scale).clamp(0.0, 1.0)

    depth_enabled = bool(calibration.get("enabled", False))
    if depth_enabled:
        scale = float(calibration["scale"])
        shift = float(calibration["shift"])
        aligned_depth = sampled_depth * scale + shift
        signed_residual = aligned_depth - geometry["camera_depth"].float()
        residual = signed_residual.abs()
        tolerance = max(
            float(calibration.get("p90_abs_residual") or 0.0),
            float(calibration.get("minimum_depth_tolerance", 0.02)),
        )
        depth_weight = torch.exp(-0.5 * (residual / tolerance).square()) * valid
        normalized_residual = (residual / tolerance).clamp(0.0, 4.0) * 0.25
        signed_normalized_residual = (
            (signed_residual / tolerance).clamp(-4.0, 4.0) * 0.25
        )
    else:
        depth_weight = valid
        normalized_residual = torch.zeros_like(valid)
        signed_normalized_residual = torch.zeros_like(valid)

    combined_weight = mask_weight * confidence_weight * depth_weight
    image_grid = geometry["image_grid"].float()
    u = (image_grid[..., 0] + 1.0) * max(float(image_width - 1), 1.0) * 0.5
    v = (image_grid[..., 1] + 1.0) * max(float(image_height - 1), 1.0) * 0.5
    ray_x = (u - intrinsics[:, None, 0, 2].float()) / intrinsics[
        :, None, 0, 0
    ].float().clamp_min(1.0e-6)
    ray_y = (v - intrinsics[:, None, 1, 2].float()) / intrinsics[
        :, None, 1, 1
    ].float().clamp_min(1.0e-6)
    ray_z = torch.full_like(ray_x, float(camera_forward_sign))
    ray = F.normalize(torch.stack((ray_x, ray_y, ray_z), dim=-1), dim=-1)
    valid_depth = geometry["camera_depth"].abs() * valid
    depth_scale = valid_depth[valid_depth > 0].median().clamp_min(1.0e-6) if bool(
        (valid_depth > 0).any().item()
    ) else valid_depth.new_tensor(1.0)
    normalized_camera_depth = (geometry["camera_depth"].abs() / depth_scale).clamp(0.0, 4.0) * 0.25

    per_view_geometry = torch.stack(
        (
            valid,
            mask_weight,
            confidence_weight,
            depth_weight,
            normalized_residual,
            combined_weight,
            ray[..., 0],
            ray[..., 1],
            ray[..., 2],
            normalized_camera_depth,
        ),
        dim=-1,
    )
    tensors = (
        sampled_visual,
        combined_weight,
        per_view_geometry,
        geometry["patch_grid"],
    )
    if not all(bool(torch.isfinite(tensor.float()).all().item()) for tensor in tensors):
        raise RuntimeError("per-view lifting evidence contains non-finite values")
    return {
        "mode": mode,
        "views": views,
        "patch_side": patch_side,
        "channels": channels,
        "volume_side": int(volume_side),
        "visual_patch_features": visual_patch_features,
        "sampled_visual": sampled_visual.permute(0, 2, 1).contiguous(),  # [V,N,C]
        "patch_grid": geometry["patch_grid"].float(),
        "image_grid": geometry["image_grid"].float(),
        "camera_depth": geometry["camera_depth"].float(),
        "valid": geometry["valid"],
        "mask_weight": mask_weight,
        "confidence_weight": confidence_weight,
        "depth_weight": depth_weight,
        "normalized_depth_residual": normalized_residual,
        # Audit-only field. It is deliberately not part of the trained
        # geometry schema, so existing checkpoints remain compatible.
        "signed_normalized_depth_residual": signed_normalized_residual,
        "base_weight": combined_weight,
        "per_view_geometry": per_view_geometry,
        "depth_enabled": depth_enabled,
        "extrinsics": variant_extrinsics,
    }


def evidence_from_sample(
    sample: dict[str, Any],
    *,
    device: torch.device,
    mode: str = "correct",
    visual_patch_features_override: torch.Tensor | None = None,
    object_to_world: torch.Tensor | None = None,
    heldout_index: int | None = None,
) -> dict[str, Any]:
    visual = (
        visual_patch_features_override
        if visual_patch_features_override is not None
        else sample["visual_patch_features"]
    )
    return sample_per_view_voxel_evidence(
        visual_patch_features=visual.to(device=device, dtype=torch.float16),
        predicted_depth=sample["predicted_depth"].to(device=device),
        depth_confidence=sample["depth_confidence"].to(device=device),
        masks=sample["masks"].to(device=device),
        intrinsics=sample["intrinsics"].to(device=device),
        extrinsics=sample["extrinsics"].to(device=device),
        calibration=sample["depth_calibration"],
        grid_transform=str(sample["grid_transform"]),
        extrinsics_type=str(sample["extrinsics_type"]),
        camera_forward_sign=float(sample["camera_forward_sign"]),
        mode=mode,
        object_to_world=object_to_world,
        heldout_index=heldout_index,
    )


def _sample_embedding_map(
    embedding_map: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    # embedding_map [V,D,H,W], grid [V,N,2] -> [V,N,D]
    sampled = F.grid_sample(
        embedding_map,
        grid[:, :, None, :],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[..., 0]
    return sampled.permute(0, 2, 1).contiguous()


def _sample_single_view_neighborhood(
    embedding_map: torch.Tensor,
    base_grid: torch.Tensor,
    *,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # embedding_map [1,D,H,W], base_grid [N,2]
    if radius < 0:
        raise ValueError("neighborhood radius must be non-negative")
    height, width = map(int, embedding_map.shape[-2:])
    offsets: list[tuple[float, float]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            offsets.append(
                (
                    2.0 * float(dx) / max(float(width - 1), 1.0),
                    2.0 * float(dy) / max(float(height - 1), 1.0),
                )
            )
    offset_tensor = base_grid.new_tensor(offsets)
    grid = base_grid[:, None, :] + offset_tensor[None, :, :]
    valid = ((grid >= -1.0) & (grid <= 1.0)).all(dim=-1).transpose(0, 1)
    sampled = F.grid_sample(
        embedding_map,
        grid[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]
    # [D,N,K] -> [K,N,D]
    return sampled.permute(2, 1, 0).contiguous(), valid


@dataclass
class HeldoutCorrespondenceOutput:
    error: torch.Tensor
    confidence_logit: torch.Tensor
    geometry_logit: torch.Tensor
    valid_mask: torch.Tensor
    source_count: torch.Tensor
    effective_views: torch.Tensor
    agreement: torch.Tensor
    disagreement: torch.Tensor
    reconstruction: torch.Tensor
    target: torch.Tensor
    pairwise_confidence: torch.Tensor
    pairwise_logit: torch.Tensor
    per_view_pairwise_confidence: torch.Tensor
    pairwise_peer_count: torch.Tensor
    final_source_weight: torch.Tensor


class LocalVoxelCorrespondence(nn.Module):
    """Same-voxel pairwise correspondence before multi-view aggregation.

    Each 16^3 voxel is processed independently.  Source-view identity is kept
    until every source view has been scored against the other source views.
    Only then are source values aggregated.  Geometry can bias pairwise
    matching, but geometry is not added to the value that reconstructs the
    held-out visual target.
    """

    def __init__(
        self,
        *,
        visual_channels: int,
        embedding_dim: int = 128,
        pairwise_dim: int = 64,
        geometry_hidden_dim: int = 48,
        score_hidden_dim: int = 192,
    ) -> None:
        super().__init__()
        self.visual_channels = int(visual_channels)
        self.embedding_dim = int(embedding_dim)
        self.pairwise_dim = int(pairwise_dim)
        self.geometry_channels = len(PER_VIEW_GEOMETRY_NAMES)
        self.geometry_hidden_dim = int(geometry_hidden_dim)
        self.score_hidden_dim = int(score_hidden_dim)
        if self.pairwise_dim <= 0:
            raise ValueError("pairwise_dim must be positive")

        self.visual_encoder = nn.Sequential(
            nn.LayerNorm(self.visual_channels),
            nn.Linear(self.visual_channels, self.embedding_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
        )
        self.pairwise_visual = nn.Sequential(
            nn.Linear(self.embedding_dim, self.pairwise_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.pairwise_dim, self.pairwise_dim, bias=False),
        )
        self.pairwise_geometry = nn.Sequential(
            nn.Linear(self.geometry_channels, geometry_hidden_dim),
            nn.SiLU(),
            nn.Linear(geometry_hidden_dim, self.pairwise_dim, bias=False),
        )
        self.value_projection = nn.Linear(
            self.embedding_dim, self.embedding_dim, bias=False
        )
        self.reconstruction = nn.Linear(
            self.embedding_dim, self.embedding_dim, bias=False
        )
        # Start visual-first.  Geometry may become useful, but cannot dominate
        # the pair score at initialization.
        self.log_visual_temperature = nn.Parameter(torch.tensor(0.0))
        self.geometry_pair_scale = nn.Parameter(torch.tensor(0.0))
        self.pairwise_bias = nn.Parameter(torch.tensor(0.0))

        summary_dim = 2 * self.geometry_channels + 4
        self.score_head = nn.Sequential(
            nn.Linear(4 * self.embedding_dim + summary_dim, score_hidden_dim),
            nn.SiLU(),
            nn.Linear(score_hidden_dim, max(score_hidden_dim // 2, 1)),
            nn.SiLU(),
            nn.Linear(max(score_hidden_dim // 2, 1), 1),
        )
        self.geometry_only_head = nn.Sequential(
            nn.Linear(summary_dim, max(32, geometry_hidden_dim)),
            nn.SiLU(),
            nn.Linear(max(32, geometry_hidden_dim), 1),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "format": CORRESPONDENCE_MODEL_VERSION,
            "visual_channels": self.visual_channels,
            "embedding_dim": self.embedding_dim,
            "pairwise_dim": self.pairwise_dim,
            "geometry_channels": self.geometry_channels,
            "geometry_hidden_dim": self.geometry_hidden_dim,
            "score_hidden_dim": self.score_hidden_dim,
            "geometry_names": list(PER_VIEW_GEOMETRY_NAMES),
            "same_voxel_only": True,
            "view_identity_preserved": True,
            "pairwise_before_aggregation": True,
            "geometry_in_value": False,
            "spatial_attention": False,
            "schema_hash": correspondence_schema_hash(),
        }

    def encode_patch_maps(self, visual_patch_features: torch.Tensor) -> torch.Tensor:
        views, patch_count, channels = map(int, visual_patch_features.shape)
        if channels != self.visual_channels:
            raise ValueError(
                f"visual channel mismatch: {channels} != {self.visual_channels}"
            )
        side = int(round(math.sqrt(patch_count)))
        if side * side != patch_count:
            raise ValueError("patch count must be square")
        encoded = self.visual_encoder(visual_patch_features.float())
        return encoded.permute(0, 2, 1).reshape(
            views, self.embedding_dim, side, side
        )

    def _pairwise_consensus(
        self,
        per_view_embedding: torch.Tensor,
        geometry: torch.Tensor,
        physical_weight: torch.Tensor,
        *,
        min_weight: float,
    ) -> dict[str, torch.Tensor]:
        """Compute learned source-source support before aggregation.

        Shapes:
          per_view_embedding [V,N,D]
          geometry           [V,N,G]
          physical_weight    [V,N], with excluded/held-out views set to zero
        """

        if per_view_embedding.ndim != 3:
            raise ValueError("per_view_embedding must be [V,N,D]")
        if geometry.shape[:2] != per_view_embedding.shape[:2]:
            raise ValueError("geometry and visual embedding shapes do not align")
        if physical_weight.shape != per_view_embedding.shape[:2]:
            raise ValueError("physical_weight must be [V,N]")

        views = int(per_view_embedding.shape[0])
        visual_pair = F.normalize(
            self.pairwise_visual(per_view_embedding), dim=-1, eps=1.0e-6
        )
        geometry_pair = F.normalize(
            self.pairwise_geometry(geometry.float()), dim=-1, eps=1.0e-6
        )
        visual_similarity = torch.einsum(
            "vnd,wnd->vwn", visual_pair, visual_pair
        )
        geometry_similarity = torch.einsum(
            "vnd,wnd->vwn", geometry_pair, geometry_pair
        )
        visual_temperature = self.log_visual_temperature.exp().clamp(0.25, 16.0)
        pairwise_logits = (
            visual_temperature * visual_similarity
            + self.geometry_pair_scale.tanh() * geometry_similarity
            + self.pairwise_bias
        )

        active = physical_weight.gt(float(min_weight))
        pair_mask = active[:, None, :] & active[None, :, :]
        eye = torch.eye(views, device=pair_mask.device, dtype=torch.bool)[:, :, None]
        pair_mask = pair_mask & ~eye
        peer_weight = physical_weight[None, :, :].expand(views, -1, -1)
        weighted_peer = peer_weight * pair_mask.float()
        denominator = weighted_peer.sum(dim=1)

        pair_probability = torch.sigmoid(pairwise_logits)
        per_view_confidence = (
            pair_probability * weighted_peer
        ).sum(dim=1) / denominator.clamp_min(1.0e-6)
        per_view_logit = (
            pairwise_logits * weighted_peer
        ).sum(dim=1) / denominator.clamp_min(1.0e-6)
        peer_count = pair_mask.sum(dim=1)
        has_peer = peer_count.gt(0)
        per_view_confidence = per_view_confidence * has_peer.float()
        per_view_logit = per_view_logit * has_peer.float()

        final_weight = physical_weight * per_view_confidence
        final_sum = final_weight.sum(dim=0)
        physical_sum = physical_weight.sum(dim=0)
        pairwise_confidence = (
            per_view_confidence * physical_weight
        ).sum(dim=0) / physical_sum.clamp_min(1.0e-6)
        pairwise_logit = (
            per_view_logit * physical_weight
        ).sum(dim=0) / physical_sum.clamp_min(1.0e-6)

        value = F.normalize(
            self.value_projection(per_view_embedding), dim=-1, eps=1.0e-6
        )
        consensus = (
            value * final_weight[..., None]
        ).sum(dim=0) / final_sum.clamp_min(1.0e-6)[:, None]
        consensus = F.normalize(consensus, dim=-1, eps=1.0e-6)
        centered = value - consensus[None]
        disagreement = (
            centered.square().mean(dim=-1) * final_weight
        ).sum(dim=0) / final_sum.clamp_min(1.0e-6)
        effective_views = final_sum.square() / final_weight.square().sum(dim=0).clamp_min(
            1.0e-6
        )

        return {
            "pairwise_logits": pairwise_logits,
            "pairwise_probability": pair_probability,
            "pairwise_mask": pair_mask,
            "per_view_confidence": per_view_confidence,
            "per_view_logit": per_view_logit,
            "peer_count": peer_count,
            "pairwise_confidence": pairwise_confidence,
            "pairwise_logit": pairwise_logit,
            "final_weight": final_weight,
            "consensus": consensus,
            "disagreement": disagreement,
            "effective_views": effective_views,
            "visual_pair_embedding": visual_pair,
            "geometry_similarity": geometry_similarity,
        }

    def evaluate_heldout(
        self,
        evidence: dict[str, Any],
        heldout_index: int,
        *,
        neighborhood_radius: int = 1,
        min_source_views: int = 2,
        min_weight: float = 1.0e-6,
        encoded_patch_maps: torch.Tensor | None = None,
        target_evidence: dict[str, Any] | None = None,
        target_encoded_patch_maps: torch.Tensor | None = None,
        detach_target: bool = True,
    ) -> HeldoutCorrespondenceOutput:
        """Reconstruct one fixed held-out target from pairwise-filtered sources."""

        views = int(evidence["views"])
        target_evidence = evidence if target_evidence is None else target_evidence
        if int(target_evidence["views"]) != views:
            raise ValueError("source and target evidence must have the same view count")
        if not 0 <= heldout_index < views:
            raise IndexError(f"heldout view {heldout_index} outside {views}")
        if views < min_source_views + 1:
            raise ValueError(
                f"need at least {min_source_views + 1} views, got {views}"
            )

        patch_maps = (
            encoded_patch_maps
            if encoded_patch_maps is not None
            else self.encode_patch_maps(evidence["visual_patch_features"])
        )
        target_patch_maps = (
            target_encoded_patch_maps
            if target_encoded_patch_maps is not None
            else self.encode_patch_maps(target_evidence["visual_patch_features"])
        )
        per_view_embedding = _sample_embedding_map(
            patch_maps, evidence["patch_grid"]
        )
        per_view_embedding = F.normalize(
            per_view_embedding, dim=-1, eps=1.0e-6
        )
        geometry = evidence["per_view_geometry"].float()
        physical_weight = evidence["base_weight"].float().clone()
        physical_weight[heldout_index] = 0.0
        source_count = physical_weight.gt(float(min_weight)).sum(dim=0)

        pairwise = self._pairwise_consensus(
            per_view_embedding,
            geometry,
            physical_weight,
            min_weight=min_weight,
        )
        reconstruction = F.normalize(
            self.reconstruction(pairwise["consensus"]), dim=-1, eps=1.0e-6
        )

        target_candidates, target_neighbor_valid = _sample_single_view_neighborhood(
            target_patch_maps[heldout_index : heldout_index + 1],
            target_evidence["patch_grid"][heldout_index],
            radius=neighborhood_radius,
        )
        target_candidates = F.normalize(
            target_candidates, dim=-1, eps=1.0e-6
        )
        if detach_target:
            target_candidates = target_candidates.detach()
        cosine = torch.einsum("nd,knd->kn", reconstruction, target_candidates)
        cosine = cosine.masked_fill(~target_neighbor_valid, -2.0)
        best_cosine, best_index = cosine.max(dim=0)
        gather_index = best_index.reshape(1, -1, 1).expand(
            1, best_index.numel(), self.embedding_dim
        )
        target = torch.gather(target_candidates, 0, gather_index)[0]
        error = (1.0 - best_cosine).clamp(0.0, 2.0)

        physical_sum = physical_weight.sum(dim=0)
        source_geometry = (
            geometry * physical_weight[..., None]
        ).sum(dim=0) / physical_sum.clamp_min(1.0e-6)[:, None]
        target_geometry = target_evidence["per_view_geometry"][
            heldout_index
        ].float()
        target_weight = target_evidence["base_weight"][heldout_index].float()
        view_denominator = max(float(views - 1), 1.0)
        summary = torch.cat(
            (
                source_geometry,
                target_geometry,
                (source_count.float() / view_denominator)[:, None],
                (pairwise["effective_views"] / view_denominator)[:, None],
                pairwise["pairwise_confidence"][:, None],
                pairwise["disagreement"][:, None],
            ),
            dim=-1,
        )
        interaction = torch.cat(
            (
                reconstruction,
                target,
                reconstruction * target,
                (reconstruction - target).abs(),
                summary,
            ),
            dim=-1,
        )
        confidence_logit = self.score_head(interaction)[:, 0]
        geometry_logit = self.geometry_only_head(summary)[:, 0]
        final_supported = pairwise["final_weight"].sum(dim=0).gt(float(min_weight))
        valid_mask = (
            source_count.ge(int(min_source_views))
            & final_supported
            & target_weight.gt(float(min_weight))
            & target_neighbor_valid.any(dim=0)
            & torch.isfinite(error)
        )
        return HeldoutCorrespondenceOutput(
            error=error,
            confidence_logit=confidence_logit,
            geometry_logit=geometry_logit,
            valid_mask=valid_mask,
            source_count=source_count,
            effective_views=pairwise["effective_views"],
            agreement=pairwise["pairwise_confidence"],
            disagreement=pairwise["disagreement"],
            reconstruction=reconstruction,
            target=target,
            pairwise_confidence=pairwise["pairwise_confidence"],
            pairwise_logit=pairwise["pairwise_logit"],
            per_view_pairwise_confidence=pairwise["per_view_confidence"],
            pairwise_peer_count=pairwise["peer_count"],
            final_source_weight=pairwise["final_weight"],
        )

    def aggregate_volume(
        self,
        evidence: dict[str, Any],
        *,
        neighborhood_radius: int = 1,
        min_source_views: int = 2,
        confidence_floor: float = 0.0,
        include_pair_features: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, float]]:
        del neighborhood_radius  # Pairwise all-view consensus has no held-out target.
        views = int(evidence["views"])
        patch_maps = self.encode_patch_maps(evidence["visual_patch_features"])
        exact_embedding = _sample_embedding_map(patch_maps, evidence["patch_grid"])
        exact_embedding = F.normalize(exact_embedding, dim=-1, eps=1.0e-6)
        base_weight = evidence["base_weight"].float()
        pairwise = self._pairwise_consensus(
            exact_embedding,
            evidence["per_view_geometry"].float(),
            base_weight,
            min_weight=1.0e-6,
        )
        per_view_confidence = pairwise["per_view_confidence"]
        final_weight = pairwise["final_weight"]
        if confidence_floor > 0.0:
            keep = per_view_confidence.ge(float(confidence_floor)).float()
            final_weight = final_weight * keep
        source_count = base_weight.gt(1.0e-6).sum(dim=0)
        supported = (
            final_weight.sum(dim=0).gt(1.0e-6)
            & source_count.ge(int(min_source_views))
        )
        sum_weight = final_weight.sum(dim=0).clamp_min(1.0e-6)
        sampled_visual = evidence["sampled_visual"].float()
        visual = (
            sampled_visual * final_weight[..., None]
        ).sum(dim=0) / sum_weight[:, None]
        visual = visual * supported[:, None]

        value = F.normalize(
            self.value_projection(exact_embedding), dim=-1, eps=1.0e-6
        )
        embedding_mean = (
            value * final_weight[..., None]
        ).sum(dim=0) / sum_weight[:, None]
        embedding_mean = F.normalize(embedding_mean, dim=-1, eps=1.0e-6)
        disagreement = (
            (value - embedding_mean[None]).square().mean(dim=-1) * final_weight
        ).sum(dim=0) / sum_weight
        effective_views = sum_weight.square() / final_weight.square().sum(dim=0).clamp_min(
            1.0e-6
        )
        confidence_map = pairwise["pairwise_confidence"] * supported.float()
        stats = {
            "supported_voxel_ratio": float(supported.float().mean().item()),
            "mean_correspondence_confidence": float(
                confidence_map[supported].mean().item()
            )
            if bool(supported.any().item())
            else 0.0,
            "mean_correspondence_error": 0.0,
            "mean_effective_views": float(effective_views[supported].mean().item())
            if bool(supported.any().item())
            else 0.0,
        }
        side = int(evidence["volume_side"])
        channels = int(evidence["channels"])
        result: dict[str, torch.Tensor | dict[str, float]] = {
            "visual_volume": visual.transpose(0, 1).reshape(
                channels, side, side, side
            ),
            "confidence": confidence_map.reshape(1, side, side, side),
            "disagreement": disagreement.reshape(1, side, side, side),
            "effective_view_fraction": (
                effective_views / max(float(views), 1.0)
            ).clamp(0.0, 1.0).reshape(1, side, side, side),
            "per_view_confidence": per_view_confidence,
            "per_view_valid": pairwise["peer_count"].gt(0),
            "per_view_error": torch.zeros_like(per_view_confidence),
            "final_weight": final_weight,
            "pairwise_logits": pairwise["pairwise_logits"],
            "pairwise_mask": pairwise["pairwise_mask"],
            "stats": stats,
        }
        if include_pair_features:
            pair_indices = torch.triu_indices(
                views,
                views,
                offset=1,
                device=base_weight.device,
            )
            left_index, right_index = pair_indices[0], pair_indices[1]
            pair_embedding = pairwise["visual_pair_embedding"]
            left = pair_embedding.index_select(0, left_index)
            right = pair_embedding.index_select(0, right_index)
            pair_logits = pairwise["pairwise_logits"][left_index, right_index]
            pair_probability = pairwise["pairwise_probability"][
                left_index, right_index
            ]
            pair_valid = pairwise["pairwise_mask"][left_index, right_index]
            left_weight = base_weight.index_select(0, left_index)
            right_weight = base_weight.index_select(0, right_index)
            left_geometry = evidence["per_view_geometry"].float().index_select(
                0, left_index
            )
            right_geometry = evidence["per_view_geometry"].float().index_select(
                0, right_index
            )
            ray_cosine = (
                left_geometry[..., 6:9] * right_geometry[..., 6:9]
            ).sum(dim=-1).clamp(-1.0, 1.0)
            geometry_similarity = pairwise["geometry_similarity"][
                left_index, right_index
            ]
            scalar_features = torch.stack(
                (
                    pair_logits,
                    pair_probability,
                    torch.minimum(left_weight, right_weight),
                    0.5 * (left_weight + right_weight),
                    (left_geometry[..., 4] - right_geometry[..., 4]).abs(),
                    ray_cosine,
                    (left_geometry[..., 9] - right_geometry[..., 9]).abs(),
                    geometry_similarity,
                ),
                dim=-1,
            )
            pair_features = torch.cat(
                (
                    0.5 * (left + right),
                    (left - right).abs(),
                    left * right,
                    scalar_features,
                ),
                dim=-1,
            )
            expected_dim = pair_feature_dim(self.pairwise_dim)
            if int(pair_features.shape[-1]) != expected_dim:
                raise RuntimeError(
                    f"pair feature dimension mismatch: {pair_features.shape[-1]} "
                    f"!= {expected_dim}"
                )
            pair_features = pair_features * pair_valid[..., None].float()
            pair_count = int(pair_features.shape[0])
            result.update(
                {
                    "pair_features": pair_features.permute(0, 2, 1).reshape(
                        pair_count, expected_dim, side, side, side
                    ).to(dtype=torch.float16),
                    "pair_valid": pair_valid.reshape(
                        pair_count, 1, side, side, side
                    ),
                    "pair_indices": pair_indices.transpose(0, 1).contiguous(),
                }
            )
            stats.update(
                {
                    "pair_feature_export_version": PAIR_FEATURE_EXPORT_VERSION,
                    "pair_feature_dim": expected_dim,
                    "pair_count": pair_count,
                    "pair_valid_ratio": float(pair_valid.float().mean().item())
                    if pair_valid.numel()
                    else 0.0,
                }
            )
        return result


def _base_metadata_from_evidence(
    evidence: dict[str, Any],
    *,
    prior_coords: torch.Tensor,
    prior_confidence: torch.Tensor,
) -> torch.Tensor:
    device = evidence["base_weight"].device
    views = int(evidence["views"])
    side = int(evidence["volume_side"])
    valid = evidence["valid"].float()
    mask_weight = evidence["mask_weight"].float()
    confidence_weight = evidence["confidence_weight"].float()
    depth_weight = evidence["depth_weight"].float()
    normalized_residual = evidence["normalized_depth_residual"].float()
    weight = evidence["base_weight"].float()
    occupancy, prior_conf, prior_distance = _prior_volume_features(
        prior_coords,
        prior_confidence,
        device=device,
        volume_side=side,
    )
    axis = (torch.arange(side, device=device, dtype=torch.float32) + 0.5)
    axis = axis / float(side) * 2.0 - 1.0
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=0).reshape(3, -1)
    view_denominator = max(float(views), 1.0)
    depth_consistency = depth_weight if bool(evidence["depth_enabled"]) else torch.zeros_like(depth_weight)
    metadata_flat = torch.cat(
        (
            (weight.sum(dim=0) / view_denominator).clamp(0.0, 1.0)[None],
            (mask_weight.sum(dim=0) / view_denominator).clamp(0.0, 1.0)[None],
            (valid.sum(dim=0) / view_denominator).clamp(0.0, 1.0)[None],
            ((depth_consistency * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1.0))[None],
            ((confidence_weight * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1.0))[None],
            ((normalized_residual * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1.0))[None],
            occupancy.reshape(1, -1),
            prior_conf.reshape(1, -1),
            prior_distance.reshape(1, -1),
            xyz,
        ),
        dim=0,
    )
    return metadata_flat.reshape(len(LIFTING_METADATA_NAMES), side, side, side)


@torch.no_grad()
def correspondence_volume_from_sample(
    sample: dict[str, Any],
    *,
    device: torch.device,
    model: LocalVoxelCorrespondence,
    mode: str = "correct",
    visual_patch_features_override: torch.Tensor | None = None,
    neighborhood_radius: int = 1,
    min_source_views: int = 2,
    confidence_floor: float = 0.0,
    object_to_world: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    evidence = evidence_from_sample(
        sample,
        device=device,
        mode=mode,
        visual_patch_features_override=visual_patch_features_override,
        object_to_world=object_to_world,
    )
    aggregated = model.aggregate_volume(
        evidence,
        neighborhood_radius=neighborhood_radius,
        min_source_views=min_source_views,
        confidence_floor=confidence_floor,
    )
    base_metadata = _base_metadata_from_evidence(
        evidence,
        prior_coords=sample["prior_coords"].to(device=device),
        prior_confidence=sample["prior_confidence"].to(device=device),
    )
    metadata = torch.cat(
        (
            base_metadata,
            aggregated["confidence"],
            aggregated["disagreement"],
            aggregated["effective_view_fraction"],
        ),
        dim=0,
    )
    if metadata.shape[0] != len(CORRESPONDENCE_METADATA_NAMES):
        raise RuntimeError("correspondence metadata schema mismatch")
    stats = dict(aggregated["stats"])
    stats.update(
        {
            "format": CORRESPONDENCE_VOLUME_VERSION,
            "mode": mode,
            "depth_enabled": bool(evidence["depth_enabled"]),
            "metadata_names": list(CORRESPONDENCE_METADATA_NAMES),
        }
    )
    return (
        aggregated["visual_volume"].unsqueeze(0),
        metadata.unsqueeze(0),
        stats,
    )


@torch.no_grad()
def correspondence_pair_volume_from_sample(
    sample: dict[str, Any],
    *,
    device: torch.device,
    model: LocalVoxelCorrespondence,
    mode: str = "correct",
    visual_patch_features_override: torch.Tensor | None = None,
    neighborhood_radius: int = 1,
    min_source_views: int = 2,
    confidence_floor: float = 0.0,
    object_to_world: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    """Build an aggregated visual volume plus unaggregated same-voxel pair tokens."""

    evidence = evidence_from_sample(
        sample,
        device=device,
        mode=mode,
        visual_patch_features_override=visual_patch_features_override,
        object_to_world=object_to_world,
    )
    aggregated = model.aggregate_volume(
        evidence,
        neighborhood_radius=neighborhood_radius,
        min_source_views=min_source_views,
        confidence_floor=confidence_floor,
        include_pair_features=True,
    )
    base_metadata = _base_metadata_from_evidence(
        evidence,
        prior_coords=sample["prior_coords"].to(device=device),
        prior_confidence=sample["prior_confidence"].to(device=device),
    )
    metadata = torch.cat(
        (
            base_metadata,
            aggregated["confidence"],
            aggregated["disagreement"],
            aggregated["effective_view_fraction"],
        ),
        dim=0,
    )
    if int(metadata.shape[0]) != len(CORRESPONDENCE_METADATA_NAMES):
        raise RuntimeError("correspondence metadata schema mismatch")
    pair_features = aggregated.get("pair_features")
    pair_valid = aggregated.get("pair_valid")
    if not torch.is_tensor(pair_features) or not torch.is_tensor(pair_valid):
        raise RuntimeError("pair feature export was not produced")
    if pair_features.ndim != 5 or pair_valid.ndim != 5:
        raise RuntimeError("pair feature tensors must be [P,C,16,16,16]")
    if pair_features.shape[0] != pair_valid.shape[0]:
        raise RuntimeError("pair feature/mask count mismatch")
    stats = dict(aggregated["stats"])
    stats.update(
        {
            "format": CORRESPONDENCE_VOLUME_VERSION,
            "pair_feature_export_version": PAIR_FEATURE_EXPORT_VERSION,
            "mode": mode,
            "depth_enabled": bool(evidence["depth_enabled"]),
            "metadata_names": list(CORRESPONDENCE_METADATA_NAMES),
            "pair_feature_scalar_names": list(PAIR_FEATURE_SCALAR_NAMES),
            "pair_indices": aggregated["pair_indices"].detach().cpu().tolist(),
        }
    )
    return (
        aggregated["visual_volume"].unsqueeze(0),
        metadata.unsqueeze(0),
        pair_features.unsqueeze(0),
        pair_valid.unsqueeze(0),
        stats,
    )


def save_correspondence_checkpoint(
    path: str | Path,
    *,
    model: LocalVoxelCorrespondence,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    args: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": CORRESPONDENCE_MODEL_VERSION,
        "step": int(step),
        "model": model.state_dict(),
        "model_metadata": model.metadata(),
        "args": dict(args),
        "history": list(history or []),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_correspondence_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    visual_channels: int | None = None,
) -> tuple[LocalVoxelCorrespondence, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("format") != CORRESPONDENCE_MODEL_VERSION:
        raise ValueError(
            f"unsupported correspondence checkpoint={checkpoint.get('format')!r}"
        )
    metadata = checkpoint.get("model_metadata", {})
    if metadata.get("schema_hash") != correspondence_schema_hash():
        raise ValueError("correspondence checkpoint schema hash mismatch")
    channels = int(metadata["visual_channels"])
    if visual_channels is not None and channels != int(visual_channels):
        raise ValueError(f"visual channel mismatch: checkpoint={channels}, cache={visual_channels}")
    model = LocalVoxelCorrespondence(
        visual_channels=channels,
        embedding_dim=int(metadata["embedding_dim"]),
        pairwise_dim=int(metadata.get("pairwise_dim", 64)),
        geometry_hidden_dim=int(metadata.get("geometry_hidden_dim", 48)),
        score_hidden_dim=int(metadata.get("score_hidden_dim", 192)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def protocol_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
