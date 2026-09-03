from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ar_ss_flow.correspondence_lifting import (
    PER_VIEW_GEOMETRY_NAMES,
    sample_per_view_voxel_evidence,
)


VIEW_IDENTITY_EVIDENCE_VERSION = (
    "pose_point_depth_mv.view_identity_evidence.v1"
)
VIEW_IDENTITY_PROBE_VERSION = "pose_point_depth_mv.view_identity_probe.v1"
VIEW_IDENTITY_CHECKPOINT_VERSION = (
    "pose_point_depth_mv.view_identity_checkpoint.v1"
)
SPATIAL_TOLERANCE_VERSION = "pose_point_depth_mv.spatial_tolerance.gaussian3.v1"
SPATIAL_TOLERANCE_MODES = ("exact", "gaussian3")
SPATIAL_TOLERANCE_DEFINITION = (
    "support-normalized separable [1,2,1]^3 aggregation; equivalent to "
    "averaging the eight half-voxel trilinear resamples"
)

VIEW_IDENTITY_CONTROL_NAMES = (
    "pose_cyclic1",
    "pose_reverse",
    "depth_view_cyclic1",
    "depth_spatial",
    "visual_view_cyclic1",
)
SPATIAL_VIEW_MISALIGNED_CONTROL = "spatial_view_misaligned"

VIEW_IDENTITY_ABLATION_MODES = (
    "full",
    "state_only",
    "visual_only",
    "geometry_only",
    "pairwise_off",
    "view_identity_off",
)

VIEW_IDENTITY_GEOMETRY_NAMES = (
    *PER_VIEW_GEOMETRY_NAMES,
    "world_ray_x",
    "world_ray_y",
    "world_ray_z",
    "camera_origin_x",
    "camera_origin_y",
    "camera_origin_z",
    "patch_u_normalized",
    "patch_v_normalized",
)


def view_identity_schema_hash() -> str:
    payload = "\n".join(
        (
            VIEW_IDENTITY_EVIDENCE_VERSION,
            *VIEW_IDENTITY_GEOMETRY_NAMES,
            *VIEW_IDENTITY_CONTROL_NAMES,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _depth_variant(
    depth: torch.Tensor,
    confidence: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "depth_view_cyclic1":
        if int(depth.shape[0]) < 2:
            return depth, confidence
        return (
            torch.roll(depth, shifts=1, dims=0),
            torch.roll(confidence, shifts=1, dims=0),
        )
    if mode == "depth_spatial":
        height, width = map(int, depth.shape[-2:])
        shifts = (max(1, height // 6), max(1, width // 5))
        return (
            torch.flip(torch.roll(depth, shifts=shifts, dims=(-2, -1)), dims=(-1,)),
            torch.flip(
                torch.roll(confidence, shifts=shifts, dims=(-2, -1)),
                dims=(-1,),
            ),
        )
    return depth, confidence


def _canonical_xyz(
    side: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    axis = (torch.arange(side, device=device, dtype=dtype) + 0.5)
    axis = axis / float(side) * 2.0 - 1.0
    return torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1
    ).reshape(-1, 3)


def build_view_identity_evidence(
    sample: dict[str, Any],
    *,
    device: torch.device,
    mode: str = "correct",
    volume_side: int = 16,
) -> dict[str, Any]:
    """Lift visual evidence without aggregating the view dimension.

    The result keeps one token for every ``(view, voxel)`` pair. Pose and
    depth controls only change their intended image/geometry binding.
    """

    allowed = {"correct", *VIEW_IDENTITY_CONTROL_NAMES}
    if mode not in allowed:
        raise ValueError(f"unsupported view-identity mode={mode!r}")

    visual = sample["visual_patch_features"].to(
        device=device, dtype=torch.float16
    )
    depth = sample["predicted_depth"].to(device=device, dtype=torch.float32)
    confidence = sample["depth_confidence"].to(
        device=device, dtype=torch.float32
    )
    masks = sample["masks"].to(device=device, dtype=torch.float32)
    intrinsics = sample["intrinsics"].to(device=device, dtype=torch.float32)
    extrinsics = sample["extrinsics"].to(device=device, dtype=torch.float32)

    evidence_mode = mode
    if mode in {"depth_view_cyclic1", "depth_spatial"}:
        depth, confidence = _depth_variant(depth, confidence, mode)
        evidence_mode = "correct"
    elif mode == "visual_view_cyclic1":
        if int(visual.shape[0]) >= 2:
            visual = torch.roll(visual, shifts=1, dims=0)
        evidence_mode = "correct"

    raw = sample_per_view_voxel_evidence(
        visual_patch_features=visual,
        predicted_depth=depth,
        depth_confidence=confidence,
        masks=masks,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        calibration=sample["depth_calibration"],
        grid_transform=str(sample["grid_transform"]),
        extrinsics_type=str(sample["extrinsics_type"]),
        camera_forward_sign=float(sample["camera_forward_sign"]),
        mode=evidence_mode,
        volume_side=int(volume_side),
    )

    variant_extrinsics = raw["extrinsics"].float()
    extrinsics_type = str(sample["extrinsics_type"])
    c2w = (
        variant_extrinsics
        if extrinsics_type == "c2w"
        else torch.linalg.inv(variant_extrinsics)
    )
    base_geometry = raw["per_view_geometry"].float()
    ray_indices = [PER_VIEW_GEOMETRY_NAMES.index(f"ray_{axis}") for axis in "xyz"]
    camera_ray = base_geometry[..., ray_indices]
    world_ray = torch.einsum(
        "vij,vnj->vni", c2w[:, :3, :3], camera_ray
    )
    world_ray = F.normalize(world_ray, dim=-1, eps=1.0e-6)

    camera_origin = c2w[:, :3, 3]
    origin_norm = torch.linalg.vector_norm(camera_origin, dim=-1)
    positive_origin_norm = origin_norm[origin_norm > 0]
    origin_scale = (
        positive_origin_norm.median().clamp_min(1.0e-6)
        if positive_origin_norm.numel()
        else camera_origin.new_tensor(1.0)
    )
    normalized_origin = (camera_origin / origin_scale).clamp(-4.0, 4.0) * 0.25
    normalized_origin = normalized_origin[:, None].expand(
        -1, int(raw["sampled_visual"].shape[1]), -1
    )
    patch_grid = raw["patch_grid"].float().clamp(-1.0, 1.0)
    geometry = torch.cat(
        (base_geometry, world_ray, normalized_origin, patch_grid), dim=-1
    )

    sampled_visual = raw["sampled_visual"].float()
    view_weight = raw["base_weight"].float().clamp(0.0, 1.0)
    tensors = (sampled_visual, geometry, view_weight)
    if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
        raise RuntimeError(f"uid={sample.get('uid')} mode={mode} non-finite evidence")
    if geometry.shape[-1] != len(VIEW_IDENTITY_GEOMETRY_NAMES):
        raise RuntimeError(
            f"geometry channel mismatch: {geometry.shape[-1]} != "
            f"{len(VIEW_IDENTITY_GEOMETRY_NAMES)}"
        )

    return {
        "format": VIEW_IDENTITY_EVIDENCE_VERSION,
        "schema_hash": view_identity_schema_hash(),
        "mode": mode,
        "views": int(sampled_visual.shape[0]),
        "volume_side": int(volume_side),
        "sampled_visual": sampled_visual,
        "geometry": geometry,
        "view_weight": view_weight,
        "canonical_xyz": _canonical_xyz(volume_side, device=device),
        "depth_enabled": bool(raw["depth_enabled"]),
        "signed_normalized_depth_residual": raw[
            "signed_normalized_depth_residual"
        ].float(),
    }


def _gaussian3_blur_voxel_tensor(values: torch.Tensor) -> torch.Tensor:
    """Apply the fixed separable [1,2,1]^3 kernel to [V,N,C] tensors."""

    if values.ndim != 3:
        raise ValueError("gaussian3 voxel blur expects [V,N,C]")
    views, voxel_count, channels = map(int, values.shape)
    side = round(voxel_count ** (1.0 / 3.0))
    if side**3 != voxel_count:
        raise ValueError(f"voxel count is not cubic: {voxel_count}")
    axis = values.new_tensor((1.0, 2.0, 1.0))
    kernel = (
        axis[:, None, None] * axis[None, :, None] * axis[None, None, :]
    )
    kernel = (kernel / kernel.sum()).reshape(1, 1, 3, 3, 3)
    volume = values.reshape(views, side, side, side, channels).permute(0, 4, 1, 2, 3)
    output = torch.empty_like(volume)
    # Chunk channels to bound temporary memory for 3072-D visual features.
    for start in range(0, channels, 128):
        stop = min(channels, start + 128)
        current = volume[:, start:stop].reshape(-1, 1, side, side, side)
        blurred = F.conv3d(current, kernel, padding=1)
        output[:, start:stop] = blurred.reshape(views, stop - start, side, side, side)
    return output.permute(0, 2, 3, 4, 1).reshape_as(values)


def apply_symmetric_spatial_tolerance(
    evidence: dict[str, Any],
    *,
    fixed_correct_weight: torch.Tensor,
    mode: str,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Apply the same local 3D aggregation to correct and corrupted evidence.

    Content is support-normalized with the original correct-branch view weight.
    The blurred support returned here must be passed as the weight override for
    every branch. This prevents a corruption from winning by changing coverage.
    """

    if mode not in SPATIAL_TOLERANCE_MODES:
        raise ValueError(f"unsupported spatial tolerance mode={mode!r}")
    fixed = fixed_correct_weight.float()
    if fixed.ndim != 2:
        raise ValueError("fixed correct weight must be [V,N]")
    if mode == "exact":
        return evidence, fixed
    visual = evidence["sampled_visual"].float()
    geometry = evidence["geometry"].float()
    if visual.shape[:2] != fixed.shape or geometry.shape[:2] != fixed.shape:
        raise ValueError("spatial tolerance evidence/support shape mismatch")

    denominator = _gaussian3_blur_voxel_tensor(fixed[..., None])[..., 0]
    denominator = denominator.clamp(0.0, 1.0)
    normalizer = denominator.clamp_min(1.0e-6)[..., None]
    smoothed_visual = _gaussian3_blur_voxel_tensor(
        visual * fixed[..., None]
    ) / normalizer
    smoothed_geometry = _gaussian3_blur_voxel_tensor(
        geometry * fixed[..., None]
    ) / normalizer
    supported = denominator.gt(1.0e-6)[..., None]
    result = dict(evidence)
    result.update(
        {
            "sampled_visual": smoothed_visual.masked_fill(~supported, 0.0),
            "geometry": smoothed_geometry.masked_fill(~supported, 0.0),
            "view_weight": denominator,
            "spatial_tolerance": mode,
            "spatial_tolerance_version": SPATIAL_TOLERANCE_VERSION,
            "spatial_tolerance_definition": SPATIAL_TOLERANCE_DEFINITION,
        }
    )
    signed_residual = evidence.get("signed_normalized_depth_residual")
    if torch.is_tensor(signed_residual):
        if signed_residual.shape != fixed.shape:
            raise ValueError("signed depth residual/support shape mismatch")
        smoothed_signed = _gaussian3_blur_voxel_tensor(
            signed_residual.float()[..., None] * fixed[..., None]
        ) / normalizer
        result["signed_normalized_depth_residual"] = smoothed_signed[
            ..., 0
        ].masked_fill(~supported[..., 0], 0.0)
    return result, denominator


def spatially_misalign_view_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Break same-voxel cross-view correspondence without changing support.

    Each view receives a distinct fixed 3D roll of its visual and geometry
    tokens. Visual/geometry binding within a view is preserved, while the
    evidence compared at a voxel comes from different canonical positions in
    different views. ``view_weight`` deliberately remains untouched so active
    support is identical to the correct branch.
    """

    visual = evidence["sampled_visual"]
    geometry = evidence["geometry"]
    if visual.ndim != 3 or geometry.ndim != 3:
        raise ValueError("spatial misalignment expects [V,N,C] evidence tensors")
    views, voxel_count, _ = map(int, visual.shape)
    if views < 2 or geometry.shape[:2] != visual.shape[:2]:
        raise ValueError("spatial misalignment requires matching evidence from >=2 views")
    side = round(voxel_count ** (1.0 / 3.0))
    if side**3 != voxel_count:
        raise ValueError(f"voxel count is not cubic: {voxel_count}")

    shifted_visual: list[torch.Tensor] = []
    shifted_geometry: list[torch.Tensor] = []
    shifts: list[tuple[int, int, int]] = []
    for view_index in range(views):
        shift = (
            1 + view_index % max(1, side - 1),
            1 + (2 * view_index + 1) % max(1, side - 1),
            1 + (3 * view_index + 2) % max(1, side - 1),
        )
        shifts.append(shift)
        shifted_visual.append(
            torch.roll(
                visual[view_index].reshape(side, side, side, -1),
                shifts=shift,
                dims=(0, 1, 2),
            ).reshape_as(visual[view_index])
        )
        shifted_geometry.append(
            torch.roll(
                geometry[view_index].reshape(side, side, side, -1),
                shifts=shift,
                dims=(0, 1, 2),
            ).reshape_as(geometry[view_index])
        )
    result = dict(evidence)
    result.update(
        {
            "mode": SPATIAL_VIEW_MISALIGNED_CONTROL,
            "sampled_visual": torch.stack(shifted_visual, dim=0),
            "geometry": torch.stack(shifted_geometry, dim=0),
            "spatial_misalignment_shifts": shifts,
        }
    )
    return result


def make_null_view_identity_evidence(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    result = dict(evidence)
    result.update(
        {
            "mode": "null",
            "sampled_visual": torch.zeros_like(evidence["sampled_visual"]),
            "geometry": torch.zeros_like(evidence["geometry"]),
            "view_weight": torch.zeros_like(evidence["view_weight"]),
        }
    )
    return result


class ViewIdentityPoseDepthProbe(nn.Module):
    """Same-voxel SS residual with view identity preserved until attention."""

    def __init__(
        self,
        *,
        visual_channels: int,
        hidden_dim: int = 96,
        pair_dim: int = 32,
        latent_channels: int = 8,
        min_views: int = 2,
    ) -> None:
        super().__init__()
        self.visual_channels = int(visual_channels)
        self.geometry_channels = len(VIEW_IDENTITY_GEOMETRY_NAMES)
        self.hidden_dim = int(hidden_dim)
        self.pair_dim = int(pair_dim)
        self.latent_channels = int(latent_channels)
        self.min_views = int(min_views)
        if min(self.hidden_dim, self.pair_dim, self.min_views) <= 0:
            raise ValueError("hidden_dim, pair_dim, and min_views must be positive")

        state_channels = 2 * self.latent_channels + 1 + 3
        self.state_projection = nn.Sequential(
            nn.Linear(state_channels, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.visual_encoder = nn.Sequential(
            nn.LayerNorm(self.visual_channels),
            nn.Linear(self.visual_channels, self.hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.geometry_encoder = nn.Sequential(
            nn.LayerNorm(self.geometry_channels),
            nn.Linear(self.geometry_channels, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.pair_projection = nn.Linear(self.hidden_dim, self.pair_dim, bias=False)
        self.log_attention_temperature = nn.Parameter(torch.tensor(0.0))
        # Keep a single zero gate at the final output. Starting this scale at
        # zero would also block gradients into the pairwise correspondence path.
        self.pair_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.fusion = nn.Sequential(
            nn.Linear(5 * self.hidden_dim, 2 * self.hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Linear(self.hidden_dim, self.latent_channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def metadata(self) -> dict[str, Any]:
        return {
            "version": VIEW_IDENTITY_PROBE_VERSION,
            "evidence_version": VIEW_IDENTITY_EVIDENCE_VERSION,
            "schema_hash": view_identity_schema_hash(),
            "fusion": "same_voxel_view_attention_then_pointwise_residual",
            "view_identity_preserved_before_aggregation": True,
            "pairwise_consistency_before_aggregation": True,
            "global_spatial_attention": False,
            "spatial_neighborhood": 1,
            "visual_channels": self.visual_channels,
            "geometry_channels": self.geometry_channels,
            "geometry_names": list(VIEW_IDENTITY_GEOMETRY_NAMES),
            "hidden_dim": self.hidden_dim,
            "pair_dim": self.pair_dim,
            "latent_channels": self.latent_channels,
            "min_views": self.min_views,
            "zero_centered_response": True,
            "zero_init_output": True,
            "time_normalization": "t_div_1000",
            "uses_pose_depth": True,
            "uses_flow_lora": False,
            "diagnostic_ablation_modes": list(VIEW_IDENTITY_ABLATION_MODES),
        }

    def support_gate(
        self,
        evidence: dict[str, Any],
        *,
        view_weight_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = (
            evidence["view_weight"]
            if view_weight_override is None
            else view_weight_override
        ).float()
        if weight.ndim != 2:
            raise ValueError("view weight must be [V,N]")
        return weight.gt(1.0e-6).sum(dim=0).ge(self.min_views).float()

    def _pairwise_consensus(
        self,
        visual_hidden: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair = F.normalize(
            self.pair_projection(visual_hidden), dim=-1, eps=1.0e-6
        )
        similarity = torch.einsum("vnd,wnd->vwn", pair, pair)
        views = int(weight.shape[0])
        active = weight.gt(1.0e-6)
        mask = active[:, None] & active[None]
        eye = torch.eye(views, device=weight.device, dtype=torch.bool)[:, :, None]
        mask = mask & ~eye
        peer_weight = weight[None].expand(views, -1, -1) * mask.float()
        denominator = peer_weight.sum(dim=1)
        temperature = self.log_attention_temperature.exp().clamp(0.25, 16.0)
        probability = torch.sigmoid(similarity * temperature)
        consensus = (probability * peer_weight).sum(dim=1) / denominator.clamp_min(
            1.0e-6
        )
        consensus = consensus * denominator.gt(0).float()
        return consensus, mask

    def prepare_evidence(
        self,
        evidence: dict[str, Any],
        *,
        view_weight_override: torch.Tensor | None = None,
        ablation_mode: str = "full",
    ) -> dict[str, Any]:
        """Encode evidence once for repeated teacher-forced states.

        Ablations keep the correct support/view-weight protocol fixed. The
        ``state_only`` is a fixed-support content-null diagnostic. Because the
        historical V1 output layer has a trainable bias, any remaining delta
        in this mode is the learned generic output bias, not physical content.
        ``view_identity_off`` retains the local weighted visual/geometry mean
        while removing per-view identity and pairwise disagreement.
        """

        if ablation_mode not in VIEW_IDENTITY_ABLATION_MODES:
            raise ValueError(
                f"unsupported view-identity ablation={ablation_mode!r}; "
                f"expected one of {VIEW_IDENTITY_ABLATION_MODES}"
            )
        if evidence.get("schema_hash") != view_identity_schema_hash():
            raise ValueError("view-identity evidence schema mismatch")

        device = next(self.parameters()).device
        sampled_visual = evidence["sampled_visual"].to(
            device=device, dtype=torch.float32
        )
        geometry = evidence["geometry"].to(device=device, dtype=torch.float32)
        weight = (
            evidence["view_weight"]
            if view_weight_override is None
            else view_weight_override
        ).to(device=device, dtype=torch.float32)
        views, voxel_count, channels = map(int, sampled_visual.shape)
        if voxel_count != 16**3 or channels != self.visual_channels:
            raise ValueError(
                f"invalid sampled visual shape {tuple(sampled_visual.shape)}"
            )
        if geometry.shape != (views, voxel_count, self.geometry_channels):
            raise ValueError(f"invalid geometry shape {tuple(geometry.shape)}")
        if weight.shape != (views, voxel_count):
            raise ValueError(f"invalid view weight shape {tuple(weight.shape)}")

        use_visual = ablation_mode not in {"state_only", "geometry_only"}
        use_geometry = ablation_mode not in {"state_only", "visual_only"}
        visual_hidden = (
            self.visual_encoder(sampled_visual)
            if use_visual
            else sampled_visual.new_zeros((views, voxel_count, self.hidden_dim))
        )
        geometry_hidden = (
            self.geometry_encoder(geometry)
            if use_geometry
            else geometry.new_zeros((views, voxel_count, self.hidden_dim))
        )
        view_token = visual_hidden + geometry_hidden
        active = weight.gt(1.0e-6)
        pairwise_enabled = ablation_mode in {"full", "visual_only"}
        if use_visual:
            consensus, _ = self._pairwise_consensus(visual_hidden, weight)
        else:
            consensus = active.float() * 0.5

        if ablation_mode == "pairwise_off":
            pairwise_enabled = False
        elif ablation_mode == "view_identity_off":
            raw_denominator = weight.sum(dim=0, keepdim=False)
            denominator = raw_denominator.clamp_min(1.0e-6)
            mean_token = (view_token * weight[..., None]).sum(dim=0)
            mean_token = mean_token / denominator[:, None]
            mean_token = mean_token * raw_denominator.gt(1.0e-6)[:, None]
            view_token = mean_token[None].expand(views, -1, -1)
            consensus = active.float() * 0.5
            pairwise_enabled = False

        xyz = evidence["canonical_xyz"].to(device=device, dtype=torch.float32)
        if xyz.shape != (voxel_count, 3):
            raise ValueError(f"invalid canonical xyz shape {tuple(xyz.shape)}")
        return {
            "format": "pose_point_depth_mv.view_identity_prepared.v1",
            "ablation_mode": ablation_mode,
            "weight": weight,
            "active": active,
            "key": self.key_projection(view_token),
            "value": self.value_projection(view_token),
            "consensus": consensus,
            "pairwise_enabled": pairwise_enabled,
            "canonical_xyz": xyz,
        }

    def forward_prepared(
        self,
        x_t: torch.Tensor,
        stock_velocity: torch.Tensor,
        t: torch.Tensor,
        prepared: dict[str, Any],
        *,
        scale: float = 1.0,
        physical_present: bool = True,
        support_gate_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x_t.shape != stock_velocity.shape:
            raise ValueError("x_t and stock_velocity must have identical shapes")
        if x_t.ndim != 5 or x_t.shape[1:] != (self.latent_channels, 16, 16, 16):
            raise ValueError("view-identity probe expects [B,8,16,16,16]")
        if int(x_t.shape[0]) != 1:
            raise ValueError("variable-view probe currently requires batch size 1")
        if prepared.get("format") != "pose_point_depth_mv.view_identity_prepared.v1":
            raise ValueError("invalid prepared view-identity evidence")

        weight = prepared["weight"]
        active = prepared["active"]
        support = (
            active.sum(dim=0).ge(self.min_views).float()
            if support_gate_override is None
            else support_gate_override.to(device=x_t.device, dtype=torch.float32)
        )
        if support.shape != (16**3,):
            raise ValueError(f"support gate must be [4096], got {tuple(support.shape)}")
        zero = torch.zeros_like(stock_velocity)
        if not physical_present or float(scale) == 0.0:
            scalar_zero = support.new_zeros(())
            return zero, {
                "delta_rms": scalar_zero,
                "delta_abs_max": scalar_zero,
                "neutral_abs_max": scalar_zero,
                "support_ratio": support.mean(),
                "view_weight_mean": weight.mean(),
                "pair_consensus": scalar_zero,
                "attention_entropy": scalar_zero,
                "attended_rms": scalar_zero,
                "centered_response_rms": scalar_zero,
            }

        voxel_count = 16**3
        xyz = prepared["canonical_xyz"]
        state = torch.cat(
            (
                x_t.float().permute(0, 2, 3, 4, 1).reshape(voxel_count, -1),
                stock_velocity.float()
                .permute(0, 2, 3, 4, 1)
                .reshape(voxel_count, -1),
                (t.float() / 1000.0).reshape(1, 1).expand(voxel_count, 1),
                xyz,
            ),
            dim=-1,
        )
        state_hidden = self.state_projection(state)
        query = self.query_projection(state_hidden)
        key = prepared["key"]
        value = prepared["value"]
        consensus = prepared["consensus"]
        logits = torch.einsum("nd,vnd->vn", query, key) / math.sqrt(
            float(self.hidden_dim)
        )
        logits = logits + weight.clamp_min(1.0e-8).log()
        if bool(prepared["pairwise_enabled"]):
            logits = logits + self.pair_logit_scale.tanh() * (consensus - 0.5)
        logits = logits.masked_fill(~active, -1.0e4)
        attention = torch.softmax(logits, dim=0) * active.float()
        attention = attention / attention.sum(dim=0, keepdim=True).clamp_min(1.0e-6)
        attended = (value * attention[..., None]).sum(dim=0)
        disagreement = (
            (value - attended[None]).abs() * attention[..., None]
        ).sum(dim=0)

        interaction = torch.cat(
            (
                state_hidden,
                attended,
                state_hidden * attended,
                (state_hidden - attended).abs(),
                disagreement,
            ),
            dim=-1,
        )
        null_interaction = torch.cat(
            (
                state_hidden,
                torch.zeros_like(attended),
                torch.zeros_like(attended),
                state_hidden.abs(),
                torch.zeros_like(disagreement),
            ),
            dim=-1,
        )
        centered = self.fusion(interaction) - self.fusion(null_interaction)
        delta_flat = self.output(centered) * support[:, None] * float(scale)
        delta = delta_flat.reshape(1, 16, 16, 16, self.latent_channels).permute(
            0, 4, 1, 2, 3
        ).contiguous()
        neutral = support.eq(0).reshape(1, 1, 16, 16, 16)
        active_attention = attention[attention > 0]
        entropy = (
            -(active_attention * active_attention.clamp_min(1.0e-8).log()).mean()
            if active_attention.numel()
            else support.new_zeros(())
        )
        stats = {
            "delta_rms": delta.float().square().mean().sqrt(),
            "delta_abs_max": delta.float().abs().max(),
            "neutral_abs_max": (
                delta.float().abs().masked_select(neutral.expand_as(delta)).max()
                if bool(neutral.any().item())
                else delta.new_zeros((), dtype=torch.float32)
            ),
            "support_ratio": support.mean(),
            "view_weight_mean": weight.mean(),
            "pair_consensus": (
                (consensus * weight).sum() / weight.sum().clamp_min(1.0e-6)
            ),
            "attention_entropy": entropy,
            "attended_rms": attended.float().square().mean().sqrt(),
            "centered_response_rms": centered.float().square().mean().sqrt(),
        }
        return delta.to(dtype=stock_velocity.dtype), stats

    def forward(
        self,
        x_t: torch.Tensor,
        stock_velocity: torch.Tensor,
        t: torch.Tensor,
        evidence: dict[str, Any],
        *,
        scale: float = 1.0,
        physical_present: bool = True,
        view_weight_override: torch.Tensor | None = None,
        support_gate_override: torch.Tensor | None = None,
        ablation_mode: str = "full",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x_t.shape != stock_velocity.shape:
            raise ValueError("x_t and stock_velocity must have identical shapes")
        if x_t.ndim != 5 or x_t.shape[1:] != (self.latent_channels, 16, 16, 16):
            raise ValueError("view-identity probe expects [B,8,16,16,16]")
        if int(x_t.shape[0]) != 1:
            raise ValueError("variable-view probe currently requires batch size 1")
        if evidence.get("schema_hash") != view_identity_schema_hash():
            raise ValueError("view-identity evidence schema mismatch")
        if ablation_mode not in VIEW_IDENTITY_ABLATION_MODES:
            raise ValueError(
                f"unsupported view-identity ablation={ablation_mode!r}"
            )
        weight = (
            evidence["view_weight"]
            if view_weight_override is None
            else view_weight_override
        ).to(device=x_t.device, dtype=torch.float32)
        support = (
            self.support_gate(evidence, view_weight_override=weight)
            if support_gate_override is None
            else support_gate_override.to(device=x_t.device, dtype=torch.float32)
        )
        if support.shape != (16**3,):
            raise ValueError(f"support gate must be [4096], got {tuple(support.shape)}")
        if not physical_present or float(scale) == 0.0:
            zero = torch.zeros_like(stock_velocity)
            scalar_zero = support.new_zeros(())
            return zero, {
                "delta_rms": scalar_zero,
                "delta_abs_max": scalar_zero,
                "neutral_abs_max": scalar_zero,
                "support_ratio": support.mean(),
                "view_weight_mean": weight.mean(),
                "pair_consensus": scalar_zero,
                "attention_entropy": scalar_zero,
                "attended_rms": scalar_zero,
                "centered_response_rms": scalar_zero,
            }
        prepared = self.prepare_evidence(
            evidence,
            view_weight_override=weight,
            ablation_mode=ablation_mode,
        )
        return self.forward_prepared(
            x_t,
            stock_velocity,
            t,
            prepared,
            scale=scale,
            support_gate_override=support,
        )


def trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def load_view_identity_probe_state(
    module: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    expected = set(module.state_dict())
    received = set(state)
    if expected != received:
        raise RuntimeError(
            "view-identity probe state mismatch: "
            f"missing={sorted(expected - received)}, "
            f"unexpected={sorted(received - expected)}"
        )
    module.load_state_dict(state, strict=True)


def protocol_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
