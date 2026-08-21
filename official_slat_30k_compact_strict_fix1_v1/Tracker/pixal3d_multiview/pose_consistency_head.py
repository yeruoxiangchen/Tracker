from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class PoseConsistencyHead(nn.Module):
    """Predict per-view/per-voxel image-pose consistency logits.

    The head is intentionally local and cheap. In ``single`` mode it scores
    each view independently. In ``pairwise`` mode it scores visible view pairs
    at each voxel, uses pair-level scores for ranking supervision, and reduces
    pair logits to centered per-view priors for the existing view-gated
    aggregator path.
    """

    def __init__(
        self,
        feature_dim: int,
        geom_dim: int = 11,
        reduced_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        min_gate: float = 0.05,
        initial_logit: float = 2.0,
        score_mode: str = "single",
        pair_weight_mode: str = "support",
        pair_weight_threshold: float = 0.05,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.geom_dim = int(geom_dim)
        self.reduced_dim = int(reduced_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.min_gate = float(min_gate)
        self.initial_logit = float(initial_logit)
        self.pair_weight_mode = str(pair_weight_mode)
        if self.pair_weight_mode not in {"support", "front_depth", "front_depth_binary"}:
            raise ValueError(f"Unknown pose consistency pair_weight_mode: {self.pair_weight_mode}")
        self.pair_weight_threshold = float(pair_weight_threshold)
        self.score_mode = str(score_mode)
        if self.score_mode not in {"single", "pairwise"}:
            raise ValueError(f"Unknown pose consistency score_mode: {self.score_mode}")

        # Extra scalar inputs:
        # cosine-to-supported-mean, support fraction, support count fraction,
        # raw support weight.
        self.extra_dim = 4
        self.feature_reduce = nn.Linear(self.feature_dim, self.reduced_dim)
        if self.score_mode == "single":
            self.score = nn.Sequential(
                nn.LayerNorm(self.reduced_dim + self.geom_dim + self.extra_dim),
                nn.Linear(self.reduced_dim + self.geom_dim + self.extra_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 1),
            )
            nn.init.zeros_(self.score[-1].weight)
            nn.init.constant_(self.score[-1].bias, self.initial_logit)
        else:
            # Pair feature:
            # |r_i-r_j|, r_i*r_j, |g_i-g_j|, cosine, min support,
            # support delta, support-count fraction.
            self.pair_extra_dim = 4
            pair_dim = self.reduced_dim * 2 + self.geom_dim + self.pair_extra_dim
            self.pair_score = nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 1),
            )
            nn.init.zeros_(self.pair_score[-1].weight)
            nn.init.constant_(self.pair_score[-1].bias, self.initial_logit)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "geom_dim": self.geom_dim,
            "reduced_dim": self.reduced_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "min_gate": self.min_gate,
            "initial_logit": self.initial_logit,
            "score_mode": self.score_mode,
            "pair_weight_mode": self.pair_weight_mode,
            "pair_weight_threshold": self.pair_weight_threshold,
        }

    def _single_view_logits(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        geom: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict]:
        view_count = int(features.shape[0])
        normed = F.normalize(features, dim=-1, eps=1e-6)
        weight_sum = weights.sum(dim=0).clamp_min(1e-6)
        mean_feature = (normed * weights[..., None]).sum(dim=0) / weight_sum[:, None]
        mean_feature = F.normalize(mean_feature, dim=-1, eps=1e-6)
        cosine_to_mean = (normed * mean_feature[None]).sum(dim=-1).clamp(-1.0, 1.0)
        cosine_to_mean = torch.where(valid, cosine_to_mean, torch.zeros_like(cosine_to_mean))

        support_fraction = (weights.sum(dim=0) / float(max(view_count, 1))).clamp(0.0, 1.0)
        support_count_fraction = (valid.float().sum(dim=0) / float(max(view_count, 1))).clamp(0.0, 1.0)
        raw_weight = weights.clamp(0.0, 1.0)
        extra = torch.stack(
            [
                cosine_to_mean,
                support_fraction[None].expand_as(weights),
                support_count_fraction[None].expand_as(weights),
                raw_weight,
            ],
            dim=-1,
        )

        reduced = self.feature_reduce(features)
        score_input = torch.cat([reduced, geom, extra], dim=-1)
        logits = self.score(score_input).squeeze(-1)
        tensors: dict[str, torch.Tensor] = {}
        stats = {"score_mode": "single"}
        return logits, tensors, stats

    def _pair_view_weight(self, weights: torch.Tensor, geom: torch.Tensor) -> tuple[torch.Tensor, dict]:
        mode = self.pair_weight_mode
        if mode == "support":
            view_weight = weights
        elif mode == "front_depth":
            # view_geom[..., 0] is the front-depth visibility weight produced by
            # sample_view_features_for_aggregation. With visibility depth enabled
            # it is zero for points behind the visual-hull front surface.
            view_weight = geom[..., 0].float().clamp_min(0.0)
        elif mode == "front_depth_binary":
            view_weight = (geom[..., 0].float() > float(self.pair_weight_threshold)).float()
        else:
            raise ValueError(f"Unknown pair_weight_mode: {mode}")

        if geom.shape[-1] >= 5 and mode.startswith("front_depth"):
            in_image = geom[..., 2].float() > 0.5
            valid_depth = geom[..., 3].float() > 0.5
            mask_hit = geom[..., 4].float() > 0.5
            valid_view = in_image & valid_depth & mask_hit & (view_weight > 0)
            view_weight = torch.where(valid_view, view_weight, torch.zeros_like(view_weight))

        stats = {
            "pair_weight_mode": mode,
            "view_weight_mean": float(view_weight.detach().mean().cpu().item()) if view_weight.numel() else 0.0,
            "view_weight_nonzero_ratio": float((view_weight > 0).float().mean().detach().cpu().item()) if view_weight.numel() else 0.0,
        }
        return view_weight, stats

    def _pairwise_view_logits(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        geom: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict]:
        view_count, voxel_count = int(features.shape[0]), int(features.shape[1])
        reduced = self.feature_reduce(features)
        normed = F.normalize(features, dim=-1, eps=1e-6)
        support_count_fraction = (valid.float().sum(dim=0) / float(max(view_count, 1))).clamp(0.0, 1.0)

        logit_sum = features.new_zeros((view_count, voxel_count), dtype=torch.float32)
        logit_count = features.new_zeros((view_count, voxel_count), dtype=torch.float32)
        logit_weight_sum = features.new_zeros((view_count, voxel_count), dtype=torch.float32)
        pair_score_sum = features.new_zeros((voxel_count,), dtype=torch.float32)
        pair_score_weight_sum = features.new_zeros((voxel_count,), dtype=torch.float32)
        pair_logits = features.new_full((view_count, view_count, voxel_count), -1.0e4, dtype=torch.float32)
        pair_weights = features.new_zeros((view_count, view_count, voxel_count), dtype=torch.float32)
        pair_valid = torch.zeros((view_count, view_count, voxel_count), device=features.device, dtype=torch.bool)
        pair_weight_threshold = max(float(self.pair_weight_threshold), 0.0)
        pair_view_weight, pair_weight_stats = self._pair_view_weight(weights, geom)

        for i in range(view_count):
            for j in range(i + 1, view_count):
                pair_weight = torch.sqrt((pair_view_weight[i] * pair_view_weight[j]).clamp_min(0.0))
                valid_ij = valid[i] & valid[j] & (pair_weight > pair_weight_threshold)
                cosine = (normed[i] * normed[j]).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
                min_weight = torch.minimum(weights[i], weights[j]).clamp(0.0, 1.0).unsqueeze(-1)
                weight_delta = (weights[i] - weights[j]).abs().clamp(0.0, 1.0).unsqueeze(-1)
                support_count = support_count_fraction.unsqueeze(-1)
                pair_input = torch.cat(
                    [
                        (reduced[i] - reduced[j]).abs(),
                        reduced[i] * reduced[j],
                        (geom[i] - geom[j]).abs(),
                        cosine,
                        min_weight,
                        weight_delta,
                        support_count,
                    ],
                    dim=-1,
                )
                logits_ij = self.pair_score(pair_input).squeeze(-1)
                logits_ij = torch.where(valid_ij, logits_ij, torch.zeros_like(logits_ij))
                pair_weight = torch.where(valid_ij, pair_weight, torch.zeros_like(pair_weight))
                pair_gate = torch.sigmoid(logits_ij)
                logit_sum[i] = logit_sum[i] + logits_ij * pair_weight
                logit_sum[j] = logit_sum[j] + logits_ij * pair_weight
                logit_count[i] = logit_count[i] + valid_ij.float()
                logit_count[j] = logit_count[j] + valid_ij.float()
                logit_weight_sum[i] = logit_weight_sum[i] + pair_weight
                logit_weight_sum[j] = logit_weight_sum[j] + pair_weight
                pair_score_sum = pair_score_sum + pair_gate * pair_weight
                pair_score_weight_sum = pair_score_weight_sum + pair_weight
                masked_logits = torch.where(valid_ij, logits_ij, torch.full_like(logits_ij, -1.0e4))
                pair_logits[i, j] = masked_logits
                pair_logits[j, i] = masked_logits
                pair_weights[i, j] = pair_weight
                pair_weights[j, i] = pair_weight
                pair_valid[i, j] = valid_ij
                pair_valid[j, i] = valid_ij

        pair_supported_views = logit_weight_sum > 0
        raw_view_logits = torch.where(pair_supported_views, logit_sum / logit_weight_sum.clamp_min(1e-6), torch.zeros_like(logit_sum))
        center_count = pair_supported_views.float().sum(dim=0).clamp_min(1.0)
        view_center = (raw_view_logits * pair_supported_views.float()).sum(dim=0) / center_count
        logits = torch.where(pair_supported_views, raw_view_logits - view_center[None], torch.zeros_like(raw_view_logits))

        pair_supported_voxels = pair_score_weight_sum > 0
        pair_voxel_score = torch.where(
            pair_supported_voxels,
            pair_score_sum / pair_score_weight_sum.clamp_min(1e-6),
            torch.zeros_like(pair_score_sum),
        )
        if pair_supported_voxels.any():
            pair_sample_score = pair_voxel_score[pair_supported_voxels].mean()
            pair_keep_ratio = pair_score_sum.sum() / pair_score_weight_sum.sum().clamp_min(1e-6)
        else:
            pair_sample_score = pair_voxel_score.mean() * 0.0
            pair_keep_ratio = pair_score_sum.sum() * 0.0

        valid_pair_logits = pair_logits[pair_valid]
        valid_pair_weights = pair_weights[pair_valid]
        valid_view_prior = logits[pair_supported_views]
        tensors = {
            "pair_match_logits": pair_logits,
            "pair_weights": pair_weights,
            "pair_valid_mask": pair_valid,
            "pair_logit_count": logit_count,
            "pair_weight_sum": logit_weight_sum,
            "pair_voxel_score": pair_voxel_score,
            "pair_sample_score": pair_sample_score,
            "pair_keep_ratio": pair_keep_ratio,
            "raw_view_logits": raw_view_logits,
            "view_prior_logits": logits,
        }
        stats = {
            "score_mode": "pairwise",
            "pair_weight_mode": self.pair_weight_mode,
            "pair_weight_threshold": pair_weight_threshold,
            "pair_valid_ratio": float(pair_valid.float().mean().detach().cpu().item()) if pair_valid.numel() else 0.0,
            "pair_count_mean": float(logit_count[valid].mean().detach().cpu().item()) if valid.any() else 0.0,
            "pair_weight_mean": float(valid_pair_weights.detach().mean().cpu().item()) if valid_pair_weights.numel() else 0.0,
            "pair_supported_voxel_ratio": float(pair_supported_voxels.float().mean().detach().cpu().item()) if pair_supported_voxels.numel() else 0.0,
            "pair_sample_score": float(pair_sample_score.detach().cpu().item()),
            "pair_keep_ratio": float(pair_keep_ratio.detach().cpu().item()),
            "pair_logit_mean": float(valid_pair_logits.detach().mean().cpu().item()) if valid_pair_logits.numel() else 0.0,
            "pair_logit_min": float(valid_pair_logits.detach().min().cpu().item()) if valid_pair_logits.numel() else 0.0,
            "pair_logit_max": float(valid_pair_logits.detach().max().cpu().item()) if valid_pair_logits.numel() else 0.0,
            "view_prior_abs_mean": float(valid_view_prior.detach().abs().mean().cpu().item()) if valid_view_prior.numel() else 0.0,
        }
        stats.update(pair_weight_stats)
        return logits, tensors, stats

    def forward(
        self,
        sampled_features: torch.Tensor,
        support_weights: torch.Tensor,
        view_geom: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        if sampled_features.ndim != 3:
            raise ValueError(f"sampled_features should be [V,N,C], got {tuple(sampled_features.shape)}")
        if support_weights.shape != sampled_features.shape[:2]:
            raise ValueError("support_weights should be [V,N] and match sampled_features")
        if view_geom.shape[:2] != sampled_features.shape[:2]:
            raise ValueError("view_geom should be [V,N,G] and match sampled_features")
        if sampled_features.shape[-1] != self.feature_dim:
            raise ValueError(f"feature dim mismatch: got {sampled_features.shape[-1]}, expected {self.feature_dim}")
        if view_geom.shape[-1] != self.geom_dim:
            raise ValueError(f"geometry dim mismatch: got {view_geom.shape[-1]}, expected {self.geom_dim}")

        features = sampled_features.float()
        weights = support_weights.float().clamp_min(0.0)
        geom = view_geom.float()
        valid = weights > 0

        weight_sum = weights.sum(dim=0).clamp_min(1e-6)
        if self.score_mode == "single":
            logits, extra_tensors, mode_stats = self._single_view_logits(features, weights, geom, valid)
        else:
            logits, extra_tensors, mode_stats = self._pairwise_view_logits(features, weights, geom, valid)

        gate01 = torch.sigmoid(logits)
        gate = self.min_gate + (1.0 - self.min_gate) * gate01
        gate = torch.where(valid, gate, torch.zeros_like(gate))
        filtered_weights = weights * gate

        view_voxel_score = (gate * weights).sum(dim=0) / weight_sum
        voxel_score = view_voxel_score
        supported = weights.sum(dim=0) > 0
        if supported.any():
            sample_score = voxel_score[supported].mean()
            keep_ratio = filtered_weights.sum() / weights.sum().clamp_min(1e-6)
        else:
            sample_score = voxel_score.mean() * 0.0
            keep_ratio = filtered_weights.sum() * 0.0
        if self.score_mode == "pairwise" and "pair_sample_score" in extra_tensors:
            voxel_score = extra_tensors["pair_voxel_score"]
            sample_score = extra_tensors["pair_sample_score"]
            keep_ratio = extra_tensors["pair_keep_ratio"]

        stats = {
            "enabled": True,
            "type": "pose_consistency",
            "score_mode": self.score_mode,
            "feature_dim": self.feature_dim,
            "geom_dim": self.geom_dim,
            "min_gate": self.min_gate,
            "sample_score": float(sample_score.detach().cpu().item()),
            "keep_ratio": float(keep_ratio.detach().cpu().item()),
            "gate_mean": float(gate[valid].detach().mean().cpu().item()) if valid.any() else 0.0,
            "gate_min": float(gate[valid].detach().min().cpu().item()) if valid.any() else 0.0,
            "gate_max": float(gate[valid].detach().max().cpu().item()) if valid.any() else 0.0,
            "valid_ratio": float(valid.float().mean().detach().cpu().item()) if valid.numel() else 0.0,
        }
        stats.update(mode_stats)
        tensors = {
            "logits": logits,
            "gate": gate,
            "filtered_weights": filtered_weights,
            "voxel_score": voxel_score,
            "sample_score": sample_score,
            "keep_ratio": keep_ratio,
            "view_voxel_score": view_voxel_score,
        }
        tensors.update(extra_tensors)
        return filtered_weights.to(dtype=support_weights.dtype), stats, tensors


def build_pose_consistency_head(
    *,
    feature_dim: int,
    geom_dim: int = 11,
    reduced_dim: int = 128,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    min_gate: float = 0.05,
    initial_logit: float = 2.0,
    score_mode: str = "single",
    pair_weight_mode: str = "support",
    pair_weight_threshold: float = 0.05,
    device: torch.device | str = "cpu",
) -> PoseConsistencyHead:
    return PoseConsistencyHead(
        feature_dim=feature_dim,
        geom_dim=geom_dim,
        reduced_dim=reduced_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        min_gate=min_gate,
        initial_logit=initial_logit,
        score_mode=score_mode,
        pair_weight_mode=pair_weight_mode,
        pair_weight_threshold=pair_weight_threshold,
    ).to(device)


def load_pose_consistency_head(path: str | Path, *, feature_dim: int | None = None, device: torch.device | str = "cpu") -> PoseConsistencyHead:
    state = torch.load(str(path), map_location="cpu")
    config = dict(state.get("pose_consistency_head_config", {}))
    if feature_dim is not None:
        config["feature_dim"] = int(feature_dim)
    if "feature_dim" not in config:
        raise ValueError(f"pose consistency checkpoint has no feature_dim config: {path}")
    head = PoseConsistencyHead(**config)
    weights = state.get("pose_consistency_head", state.get("model"))
    if weights is None:
        raise ValueError(f"pose consistency checkpoint has no pose_consistency_head/model state: {path}")
    head.load_state_dict(weights, strict=True)
    return head.to(device)
