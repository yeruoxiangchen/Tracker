from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from pose_aligned_reconstruction.view_identity_lifting import (
    VIEW_IDENTITY_EVIDENCE_VERSION,
    VIEW_IDENTITY_GEOMETRY_NAMES,
    view_identity_schema_hash,
)


CORRESPONDENCE_HEAD_VERSION = (
    "pose_point_depth_mv.view_correspondence_head.v1"
)
CORRESPONDENCE_CHECKPOINT_VERSION = (
    "pose_point_depth_mv.view_correspondence_checkpoint.v1"
)
VOXEL_SELFCAL_VERSION = "pose_point_depth_mv.voxel_selfcal.v2"
VOXEL_CONTROL_RANKING_VERSION = (
    "pose_point_depth_mv.voxel_control_ranking.v1"
)
VOXEL_RELIABILITY_WEIGHTING_VERSION = (
    "pose_point_depth_mv.correct_voxel_reliability.v1"
)
HARD_ADMITTED_SOFT_WEIGHT_VERSION = (
    "pose_point_depth_mv.hard_admitted_soft_weight.v1"
)
CONTINUOUS_SOFT_WEIGHT_VERSION = (
    "pose_point_depth_mv.all_active_continuous_soft_weight.v1"
)
# Compatibility name for old checkpoints and external imports. New reports use
# the explicit hard-admitted name because non-positive margins are zeroed.
SOFT_VOXEL_GATE_VERSION = HARD_ADMITTED_SOFT_WEIGHT_VERSION
VOXEL_RELIABILITY_COMPONENTS = (
    "view_support_fraction",
    "pair_support_quality",
    "depth_reliability",
    "visibility_agreement",
)
DEFAULT_TRAIN_CONTROLS = (
    "pose_cyclic1",
    "depth_view_cyclic1",
    "visual_view_cyclic1",
)
DEFAULT_HELDOUT_CONTROLS = (
    "pose_reverse",
    "depth_spatial",
)


def parse_control_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not names or len(set(names)) != len(names):
        raise ValueError("control names must be non-empty and unique")
    return names


def correspondence_protocol_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def correspondence_architecture_hash(metadata: dict[str, Any]) -> str:
    """Stable architecture identity stored with C0.1 evaluation reports."""

    return correspondence_protocol_hash(metadata)


class ViewCorrespondenceHead(nn.Module):
    """Score image-pose-depth correspondence before any SS Flow fusion.

    Every view keeps its visual/geometry binding until symmetric view-pair
    features are formed. A fixed correct-branch weight can be supplied to all
    corruptions, so score differences cannot come from output support changes.
    """

    def __init__(
        self,
        *,
        visual_channels: int,
        hidden_dim: int = 64,
        pair_hidden_dim: int = 96,
        min_views: int = 2,
    ) -> None:
        super().__init__()
        self.visual_channels = int(visual_channels)
        self.geometry_channels = len(VIEW_IDENTITY_GEOMETRY_NAMES)
        self.hidden_dim = int(hidden_dim)
        self.pair_hidden_dim = int(pair_hidden_dim)
        self.min_views = int(min_views)
        if min(self.visual_channels, self.hidden_dim, self.pair_hidden_dim) <= 0:
            raise ValueError("channel and hidden dimensions must be positive")
        if self.min_views < 2:
            raise ValueError("correspondence scoring requires at least two views")

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
        self.joint_encoder = nn.Sequential(
            nn.LayerNorm(4 * self.hidden_dim),
            nn.Linear(4 * self.hidden_dim, 2 * self.hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(5 * self.hidden_dim),
            nn.Linear(5 * self.hidden_dim, self.pair_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.pair_hidden_dim, self.pair_hidden_dim),
            nn.SiLU(),
        )
        self.pair_score = nn.Linear(self.pair_hidden_dim, 1)

    def metadata(self) -> dict[str, Any]:
        return {
            "version": CORRESPONDENCE_HEAD_VERSION,
            "evidence_version": VIEW_IDENTITY_EVIDENCE_VERSION,
            "evidence_schema_hash": view_identity_schema_hash(),
            "visual_channels": self.visual_channels,
            "geometry_channels": self.geometry_channels,
            "geometry_names": list(VIEW_IDENTITY_GEOMETRY_NAMES),
            "hidden_dim": self.hidden_dim,
            "pair_hidden_dim": self.pair_hidden_dim,
            "min_views": self.min_views,
            "uses_flow_state": False,
            "uses_target_latent": False,
            "uses_ss_flow": False,
            "fixed_correct_support_protocol": True,
            "view_pair_pooling": "symmetric_same_voxel_pairs",
            "global_spatial_attention": False,
        }

    def forward(
        self,
        evidence: dict[str, Any],
        *,
        view_weight_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if evidence.get("schema_hash") != view_identity_schema_hash():
            raise ValueError("view correspondence evidence schema mismatch")
        device = next(self.parameters()).device
        visual = evidence["sampled_visual"].to(device=device, dtype=torch.float32)
        geometry = evidence["geometry"].to(device=device, dtype=torch.float32)
        weight_source = (
            evidence["view_weight"]
            if view_weight_override is None
            else view_weight_override
        )
        weight = weight_source.to(device=device, dtype=torch.float32)
        if visual.ndim != 3:
            raise ValueError("sampled visual must be [V,N,C]")
        views, voxel_count, channels = map(int, visual.shape)
        if views < self.min_views or channels != self.visual_channels:
            raise ValueError(f"invalid visual shape {tuple(visual.shape)}")
        if geometry.shape != (views, voxel_count, self.geometry_channels):
            raise ValueError(f"invalid geometry shape {tuple(geometry.shape)}")
        if weight.shape != (views, voxel_count):
            raise ValueError(f"invalid weight shape {tuple(weight.shape)}")
        if not all(
            bool(torch.isfinite(value).all().item())
            for value in (visual, geometry, weight)
        ):
            raise ValueError("non-finite correspondence input")

        visual_hidden = self.visual_encoder(visual)
        geometry_hidden = self.geometry_encoder(geometry)
        joint_hidden = self.joint_encoder(
            torch.cat(
                (
                    visual_hidden,
                    geometry_hidden,
                    visual_hidden * geometry_hidden,
                    (visual_hidden - geometry_hidden).abs(),
                ),
                dim=-1,
            )
        )

        pair_logits: list[torch.Tensor] = []
        pair_weights: list[torch.Tensor] = []
        for first, second in combinations(range(views), 2):
            first_joint = joint_hidden[first]
            second_joint = joint_hidden[second]
            features = torch.cat(
                (
                    0.5 * (first_joint + second_joint),
                    first_joint * second_joint,
                    (first_joint - second_joint).abs(),
                    visual_hidden[first] * visual_hidden[second],
                    geometry_hidden[first] * geometry_hidden[second],
                ),
                dim=-1,
            )
            pair_logits.append(self.pair_score(self.pair_encoder(features)).squeeze(-1))
            pair_weights.append(torch.minimum(weight[first], weight[second]))

        logits = torch.stack(pair_logits, dim=0)
        weights = torch.stack(pair_weights, dim=0).clamp_min(0.0)
        denominator = weights.sum(dim=0)
        active_view_count = weight.gt(1.0e-6).sum(dim=0)
        active = active_view_count.ge(self.min_views) & denominator.gt(1.0e-6)
        if not bool(active.any().item()):
            raise RuntimeError("correspondence evidence has no fixed multi-view support")
        voxel_score = (logits * weights).sum(dim=0) / denominator.clamp_min(1.0e-6)
        sample_weight = denominator * active.float()
        sample_score = (voxel_score * sample_weight).sum() / sample_weight.sum().clamp_min(
            1.0e-6
        )
        active_scores = voxel_score[active]
        return {
            "sample_score": sample_score,
            "voxel_score": voxel_score,
            "active_mask": active,
            "support_ratio": active.float().mean(),
            "pair_weight_mean": weights.mean(),
            "active_score_mean": active_scores.mean(),
            "active_score_std": active_scores.float().std(unbiased=False),
            "active_positive_ratio": active_scores.gt(0).float().mean(),
            "pair_count": sample_score.new_tensor(float(len(pair_logits))),
        }


def voxel_self_calibration(
    correct: dict[str, torch.Tensor],
    training_controls: dict[str, dict[str, torch.Tensor]],
    *,
    threshold: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Build a local gate from paired, fixed-support correspondence scores.

    The hard margin is the primary signal: a voxel is positive only when its
    correct score exceeds every synthetic corruption used during training.
    No object-level score is broadcast back into the spatial map.
    """

    if not training_controls:
        raise ValueError("voxel self-calibration requires training controls")
    correct_score = correct["voxel_score"].float()
    active = correct["active_mask"].bool()
    if correct_score.ndim != 1 or active.shape != correct_score.shape:
        raise ValueError("correct voxel score and active mask must be [N]")
    control_scores: list[torch.Tensor] = []
    for name, result in training_controls.items():
        score = result["voxel_score"].float()
        control_active = result["active_mask"].bool()
        if score.shape != correct_score.shape:
            raise ValueError(f"control={name} voxel score shape mismatch")
        if not torch.equal(active, control_active):
            raise ValueError(f"control={name} does not use fixed correct support")
        control_scores.append(score)
    stacked = torch.stack(control_scores, dim=0)
    mean_control_score = stacked.mean(dim=0)
    hard_control_score, hard_control_index = stacked.max(dim=0)
    raw_mean_margin = correct_score - mean_control_score
    raw_hard_margin = correct_score - hard_control_score
    # Never leave an attractive but meaningless margin in unsupported voxels.
    # C1 consumes the masked maps only; raw maps remain available for audit.
    mean_margin = raw_mean_margin.masked_fill(~active, 0.0)
    hard_margin = raw_hard_margin.masked_fill(~active, 0.0)
    hard_control_index = hard_control_index.masked_fill(~active, -1)
    gate = active & hard_margin.gt(float(threshold))
    active_count = active.float().sum().clamp_min(1.0)
    return {
        "correct_score": correct_score,
        "mean_control_score": mean_control_score,
        "hard_control_score": hard_control_score,
        "hard_control_index": hard_control_index,
        "raw_mean_margin": raw_mean_margin,
        "raw_hard_margin": raw_hard_margin,
        "mean_margin": mean_margin,
        "hard_margin": hard_margin,
        "active_mask": active,
        "gate_mask": gate,
        "active_ratio": active.float().mean(),
        "gate_ratio": gate.float().mean(),
        "gate_fraction_of_active": gate.float().sum() / active_count,
    }


def hard_admitted_soft_weight(
    hard_margin: torch.Tensor,
    reliability: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    temperature: float,
    reliability_power: float = 1.0,
) -> torch.Tensor:
    """Convert an admitted positive local margin into a bounded C1 weight.

    Confidence is zero for unsupported or non-positive-margin voxels.  Positive
    margins use a smooth ``tanh`` response and are modulated by detached
    correct-branch reliability.  This is an audit/export quantity, not a
    calibrated probability and not a replacement for the formal hard C0 gate.
    """

    margin = hard_margin.detach().float()
    weight = reliability.detach().float()
    active = active_mask.detach().bool()
    if margin.shape != weight.shape or margin.shape != active.shape:
        raise ValueError("hard-admitted weight inputs must have identical [N] shapes")
    if margin.ndim != 1:
        raise ValueError("hard-admitted weight inputs must be one-dimensional")
    if float(temperature) <= 0.0:
        raise ValueError("hard-admitted weight temperature must be positive")
    if float(reliability_power) < 0.0:
        raise ValueError("hard-admitted reliability power must be non-negative")
    if not all(
        bool(torch.isfinite(value).all().item()) for value in (margin, weight)
    ):
        raise ValueError("hard-admitted weight inputs must be finite")
    if bool(((weight < 0.0) | (weight > 1.0)).any().item()):
        raise ValueError("hard-admitted reliability must be in [0,1]")

    margin_confidence = torch.tanh(
        margin.clamp_min(0.0) / float(temperature)
    )
    reliability_confidence = weight.pow(float(reliability_power))
    confidence = margin_confidence * reliability_confidence
    return confidence.masked_fill(~active, 0.0).clamp(0.0, 1.0)


def continuous_voxel_gate_weight(
    hard_margin: torch.Tensor,
    reliability: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    temperature: float,
    reliability_power: float = 1.0,
    max_scale: float = 0.1,
) -> torch.Tensor:
    """Build a low-amplitude all-active C1 ablation weight.

    Unlike the formal hard-admitted weight, this sigmoid response remains
    positive for active voxels with non-positive margins. It is an ablation,
    not a calibrated probability and not part of the N3 admission protocol.
    """

    margin = hard_margin.detach().float()
    reliability = reliability.detach().float()
    active = active_mask.detach().bool()
    if margin.shape != reliability.shape or margin.shape != active.shape:
        raise ValueError("continuous weight inputs must have identical [N] shapes")
    if margin.ndim != 1:
        raise ValueError("continuous weight inputs must be one-dimensional")
    if float(temperature) <= 0.0:
        raise ValueError("continuous weight temperature must be positive")
    if float(reliability_power) < 0.0:
        raise ValueError("continuous reliability power must be non-negative")
    if not 0.0 < float(max_scale) <= 1.0:
        raise ValueError("continuous weight max_scale must be in (0,1]")
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (margin, reliability)
    ):
        raise ValueError("continuous weight inputs must be finite")
    if bool(((reliability < 0.0) | (reliability > 1.0)).any().item()):
        raise ValueError("continuous reliability must be in [0,1]")

    weight = (
        float(max_scale)
        * torch.sigmoid(margin / float(temperature))
        * reliability.pow(float(reliability_power))
    )
    return weight.masked_fill(~active, 0.0).clamp(0.0, float(max_scale))


def soft_voxel_gate_confidence(
    hard_margin: torch.Tensor,
    reliability: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    temperature: float,
    reliability_power: float = 1.0,
) -> torch.Tensor:
    """Compatibility alias for :func:`hard_admitted_soft_weight`."""

    return hard_admitted_soft_weight(
        hard_margin,
        reliability,
        active_mask,
        temperature=temperature,
        reliability_power=reliability_power,
    )


def voxel_control_ranking_loss(
    correct_score: torch.Tensor,
    control_scores: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    margin: float,
    temperature: float,
    hard_weight: float,
    voxel_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Train the same per-voxel control ordering audited by C0.1.

    The all-control term prevents non-maximum controls from losing their
    gradients.  The smooth-hard term is ``log(1 + sum(exp(violation / T)))``;
    it emphasizes the largest violation while remaining differentiable with
    respect to every control.  Exact hard margins are returned for diagnostics
    and still use the formal ``correct - max(control)`` definition.
    """

    correct = correct_score.float()
    controls = control_scores.float()
    active = active_mask.bool()
    if correct.ndim != 1:
        raise ValueError("correct voxel score must be [N]")
    if controls.ndim != 2 or controls.shape[1:] != correct.shape:
        raise ValueError("control voxel scores must be [K,N]")
    if int(controls.shape[0]) <= 0:
        raise ValueError("at least one voxel control is required")
    if active.shape != correct.shape or not bool(active.any().item()):
        raise ValueError("active voxel mask must be non-empty [N]")
    if float(temperature) <= 0.0:
        raise ValueError("voxel rank temperature must be positive")
    if float(margin) < 0.0 or float(hard_weight) < 0.0:
        raise ValueError("voxel rank margin and hard weight must be non-negative")
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (correct, controls)
    ):
        raise ValueError("voxel ranking scores must be finite")

    if voxel_weight is None:
        selected_weight = correct.new_ones(int(active.sum().item()))
    else:
        weight = voxel_weight.detach().float()
        if weight.shape != correct.shape:
            raise ValueError("voxel rank weight must be [N]")
        if not bool(torch.isfinite(weight).all().item()) or bool((weight < 0).any().item()):
            raise ValueError("voxel rank weight must be finite and non-negative")
        selected_weight = weight[active]
    weight_sum = selected_weight.sum()
    if float(weight_sum.detach().item()) <= 0.0:
        raise ValueError("active voxel rank weight must have positive mass")

    selected_correct = correct[active]
    selected_controls = controls[:, active]
    control_margins = selected_correct.unsqueeze(0) - selected_controls
    violations = float(margin) - control_margins
    scaled_violations = violations / float(temperature)

    per_control_loss_map = F.softplus(scaled_violations) * float(temperature)
    per_control_losses = (
        per_control_loss_map * selected_weight.unsqueeze(0)
    ).sum(dim=1) / weight_sum
    all_control_loss = per_control_losses.mean()

    zero_baseline = scaled_violations.new_zeros(
        (1, int(scaled_violations.shape[1]))
    )
    smooth_hard_loss_map = float(temperature) * torch.logsumexp(
        torch.cat((zero_baseline, scaled_violations), dim=0), dim=0
    )
    smooth_hard_loss = (smooth_hard_loss_map * selected_weight).sum() / weight_sum
    total_loss = all_control_loss + float(hard_weight) * smooth_hard_loss

    exact_hard_margin = control_margins.min(dim=0).values
    weighted_control_margin_means = (
        control_margins * selected_weight.unsqueeze(0)
    ).sum(dim=1) / weight_sum
    weighted_hard_margin_mean = (
        exact_hard_margin * selected_weight
    ).sum() / weight_sum
    weighted_hard_positive_ratio = (
        exact_hard_margin.gt(0).float() * selected_weight
    ).sum() / weight_sum
    return {
        "loss": total_loss,
        "all_control_loss": all_control_loss,
        "smooth_hard_loss": smooth_hard_loss,
        "per_control_losses": per_control_losses,
        "control_margin_means": weighted_control_margin_means,
        "hard_margin_mean": weighted_hard_margin_mean,
        "hard_positive_ratio": weighted_hard_positive_ratio,
        "exact_hard_margin": exact_hard_margin,
        "active_weight_sum": weight_sum,
    }


def correct_voxel_reliability_weight(
    evidence: dict[str, Any],
    active_mask: torch.Tensor,
    *,
    min_views: int,
    floor: float,
    power: float,
) -> dict[str, torch.Tensor]:
    """Build a detached confidence weight from the correct branch only.

    The returned map is shared by the correct score and every corruption in
    ``voxel_control_ranking_loss``. It therefore changes the importance of a
    fixed voxel, never the branch-specific support or output coverage.
    """

    if evidence.get("schema_hash") != view_identity_schema_hash():
        raise ValueError("voxel reliability evidence schema mismatch")
    if int(min_views) < 2:
        raise ValueError("voxel reliability requires min_views >= 2")
    if not 0.0 <= float(floor) < 1.0:
        raise ValueError("voxel reliability floor must be in [0,1)")
    if float(power) <= 0.0:
        raise ValueError("voxel reliability power must be positive")

    view_weight = evidence["view_weight"].detach().float().clamp(0.0, 1.0)
    geometry = evidence["geometry"].detach().float()
    active = active_mask.detach().bool()
    if view_weight.ndim != 2:
        raise ValueError("voxel reliability view weight must be [V,N]")
    views, voxel_count = map(int, view_weight.shape)
    if views < int(min_views) or active.shape != (voxel_count,):
        raise ValueError("voxel reliability input shape mismatch")
    if geometry.shape != (views, voxel_count, len(VIEW_IDENTITY_GEOMETRY_NAMES)):
        raise ValueError("voxel reliability geometry shape mismatch")
    if not bool(active.any().item()):
        raise ValueError("voxel reliability active mask is empty")
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (view_weight, geometry)
    ):
        raise ValueError("voxel reliability input contains non-finite values")

    geometry_index = {
        name: VIEW_IDENTITY_GEOMETRY_NAMES.index(name)
        for name in (
            "valid",
            "mask_weight",
            "depth_confidence_weight",
            "depth_consistency_weight",
        )
    }
    valid = geometry[..., geometry_index["valid"]].clamp(0.0, 1.0)
    mask = geometry[..., geometry_index["mask_weight"]].clamp(0.0, 1.0)
    depth_confidence = geometry[
        ..., geometry_index["depth_confidence_weight"]
    ].clamp(0.0, 1.0)
    depth_consistency = geometry[
        ..., geometry_index["depth_consistency_weight"]
    ].clamp(0.0, 1.0)

    supported_view = view_weight.gt(1.0e-6)
    view_support_fraction = supported_view.float().mean(dim=0)

    pair_quality_terms: list[torch.Tensor] = []
    pair_active_terms: list[torch.Tensor] = []
    for first, second in combinations(range(views), 2):
        pair_quality_terms.append(torch.minimum(view_weight[first], view_weight[second]))
        pair_active_terms.append(supported_view[first] & supported_view[second])
    pair_quality = torch.stack(pair_quality_terms, dim=0)
    pair_active = torch.stack(pair_active_terms, dim=0)
    pair_support_quality = (
        pair_quality * pair_active.float()
    ).sum(dim=0) / pair_active.float().sum(dim=0).clamp_min(1.0)

    projection_support = (valid * mask).clamp(0.0, 1.0)
    projection_mass = projection_support.sum(dim=0).clamp_min(1.0e-6)
    depth_confidence_mean = (
        depth_confidence * projection_support
    ).sum(dim=0) / projection_mass
    depth_consistency_mean = (
        depth_consistency * projection_support
    ).sum(dim=0) / projection_mass
    depth_reliability = (
        depth_confidence_mean * depth_consistency_mean
    ).clamp_min(0.0).sqrt()
    visibility_agreement = (mask * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(
        1.0e-6
    )

    components = (
        view_support_fraction,
        pair_support_quality,
        depth_reliability,
        visibility_agreement,
    )
    raw_weight = torch.stack(components, dim=0).clamp(0.0, 1.0).prod(dim=0)
    raw_weight = raw_weight.clamp_min(0.0).pow(1.0 / len(components))
    reliability = float(floor) + (1.0 - float(floor)) * raw_weight.pow(
        float(power)
    )
    reliability = reliability.masked_fill(~active, 0.0).detach()
    selected = reliability[active]
    if float(selected.sum().item()) <= 0.0:
        raise ValueError("voxel reliability has no active weight mass")
    effective_fraction = selected.sum().square() / (
        float(selected.numel()) * selected.square().sum().clamp_min(1.0e-12)
    )
    return {
        "weight": reliability,
        "raw_weight": raw_weight.masked_fill(~active, 0.0).detach(),
        "view_support_fraction": view_support_fraction.detach(),
        "pair_support_quality": pair_support_quality.detach(),
        "depth_reliability": depth_reliability.detach(),
        "depth_confidence_mean": depth_confidence_mean.detach(),
        "depth_consistency_mean": depth_consistency_mean.detach(),
        "visibility_agreement": visibility_agreement.detach(),
        "active_weight_mean": selected.mean(),
        "active_weight_min": selected.min(),
        "active_weight_max": selected.max(),
        "effective_fraction": effective_fraction,
    }


def trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def load_correspondence_head_state(
    module: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    expected = set(module.state_dict())
    received = set(state)
    if expected != received:
        raise RuntimeError(
            "correspondence head state mismatch: "
            f"missing={sorted(expected - received)}, "
            f"unexpected={sorted(received - expected)}"
        )
    module.load_state_dict(state, strict=True)
