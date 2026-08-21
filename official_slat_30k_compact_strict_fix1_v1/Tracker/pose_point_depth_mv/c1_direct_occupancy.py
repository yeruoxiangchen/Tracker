from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from pose_point_depth_mv.c1_matched_budget import (
    CORRUPTION_POLICY_NAMES,
    matched_candidate_weights,
)
from pose_point_depth_mv.c1_occupancy import (
    C1MapTargetDataset,
    balanced_binary_loss,
    file_sha256,
    load_json,
)
from pose_point_depth_mv.correspondence_head import (
    CORRESPONDENCE_CHECKPOINT_VERSION,
    ViewCorrespondenceHead,
    load_correspondence_head_state,
)
from pose_point_depth_mv.view_identity_lifting import (
    apply_symmetric_spatial_tolerance,
    build_view_identity_evidence,
)


C1_DIRECT_OCCUPANCY_VERSION = "pose_point_depth_mv.c1_direct_occupancy.v1"
C1_DIRECT_OCCUPANCY_CHECKPOINT_VERSION = (
    "pose_point_depth_mv.c1_direct_occupancy_checkpoint.v1"
)
C1_DIRECT_OCCUPANCY_EVAL_VERSION = (
    "pose_point_depth_mv.c1_direct_occupancy_eval.v1"
)
C1_DIRECT_OCCUPANCY_SUMMARY_VERSION = (
    "pose_point_depth_mv.c1_direct_occupancy_summary.v1"
)

DIRECT_MODELS = ("M0_reliability", "M1_view_geometry", "M2_plus_correspondence")
DIRECT_BRANCHES = ("correct", *CORRUPTION_POLICY_NAMES)


def protocol_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ReliabilityOccupancyProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, reliability: torch.Tensor) -> torch.Tensor:
        if reliability.ndim != 2 or reliability.shape[-1] != 1:
            raise ValueError("M0 reliability input must be [N,1]")
        return self.linear(reliability.float()).squeeze(-1)


class ViewGeometryOccupancyProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.base = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, base_features: torch.Tensor) -> torch.Tensor:
        if base_features.ndim != 2 or base_features.shape[-1] != self.input_dim:
            raise ValueError("M1 base feature shape mismatch")
        return self.base(base_features.float()).squeeze(-1)


class CorrespondenceAugmentedOccupancyProbe(ViewGeometryOccupancyProbe):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__(input_dim, hidden_dim)
        corr_hidden = max(8, int(hidden_dim) // 2)
        self.correspondence = nn.Sequential(
            nn.Linear(1, corr_hidden),
            nn.SiLU(),
            nn.Linear(corr_hidden, 1),
        )
        nn.init.zeros_(self.correspondence[-1].weight)
        nn.init.zeros_(self.correspondence[-1].bias)

    def forward(
        self, base_features: torch.Tensor, correspondence: torch.Tensor
    ) -> torch.Tensor:
        if correspondence.ndim != 2 or correspondence.shape[-1] != 1:
            raise ValueError("M2 correspondence input must be [N,1]")
        base = super().forward(base_features)
        residual = self.correspondence(correspondence.float()).squeeze(-1)
        return base + residual


def initialize_nested_models(
    *, input_dim: int, hidden_dim: int, seed: int
) -> dict[str, nn.Module]:
    torch.manual_seed(int(seed))
    m0 = ReliabilityOccupancyProbe()
    torch.manual_seed(int(seed) + 1)
    m1 = ViewGeometryOccupancyProbe(input_dim, hidden_dim)
    torch.manual_seed(int(seed) + 1)
    m2 = CorrespondenceAugmentedOccupancyProbe(input_dim, hidden_dim)
    m2.base.load_state_dict(m1.base.state_dict(), strict=True)
    return {
        "M0_reliability": m0,
        "M1_view_geometry": m1,
        "M2_plus_correspondence": m2,
    }


def nested_base_initialization_equal(models: dict[str, nn.Module]) -> bool:
    first = models["M1_view_geometry"].base.state_dict()
    second = models["M2_plus_correspondence"].base.state_dict()
    return all(torch.equal(first[name], second[name]) for name in first)


def _amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.cuda.amp.autocast(dtype=dtype)


def load_frozen_correspondence_head(
    dataset: C1MapTargetDataset,
    *,
    device: torch.device,
) -> tuple[ViewCorrespondenceHead, dict[str, Any], str]:
    checkpoint_path = Path(dataset.report["checkpoint"]).resolve()
    if file_sha256(checkpoint_path) != dataset.report["checkpoint_sha256"]:
        raise ValueError("C0 checkpoint SHA-256 differs from report")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != CORRESPONDENCE_CHECKPOINT_VERSION:
        raise ValueError("unexpected C0 checkpoint format")
    saved_args = checkpoint["args"]
    head = ViewCorrespondenceHead(
        visual_channels=dataset.cache.visual_feature_dim,
        hidden_dim=int(saved_args["hidden_dim"]),
        pair_hidden_dim=int(saved_args["pair_hidden_dim"]),
        min_views=int(saved_args["min_views"]),
    ).to(device).eval()
    load_correspondence_head_state(head, checkpoint["model_trainable_state"])
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    if head.metadata() != checkpoint["model_summary"]["head"]:
        raise ValueError("runtime C0 head metadata differs from checkpoint")
    spatial_tolerance = str(
        checkpoint["model_summary"]["protocol"].get(
            "training_spatial_tolerance", saved_args.get("spatial_tolerance", "exact")
        )
    )
    if spatial_tolerance != "gaussian3":
        raise ValueError("C1.1b requires the admitted gaussian3 head")
    return head, checkpoint, str(saved_args.get("amp_dtype", "bf16"))


def pooled_view_geometry_features(
    head: ViewCorrespondenceHead,
    evidence: dict[str, Any],
    fixed_weight: torch.Tensor,
) -> torch.Tensor:
    """Pool frozen per-view joint tokens without discarding their variance."""

    device = next(head.parameters()).device
    visual = evidence["sampled_visual"].to(device=device, dtype=torch.float32)
    geometry = evidence["geometry"].to(device=device, dtype=torch.float32)
    weight = fixed_weight.to(device=device, dtype=torch.float32).clamp_min(0.0)
    visual_hidden = head.visual_encoder(visual)
    geometry_hidden = head.geometry_encoder(geometry)
    joint = head.joint_encoder(
        torch.cat(
            (
                visual_hidden,
                geometry_hidden,
                visual_hidden * geometry_hidden,
                (visual_hidden - geometry_hidden).abs(),
            ),
            dim=-1,
        )
    ).float()
    denominator = weight.sum(dim=0).clamp_min(1.0e-6)
    normalized = weight / denominator.unsqueeze(0)
    pooled_mean = (joint * normalized.unsqueeze(-1)).sum(dim=0)
    pooled_variance = (
        (joint - pooled_mean.unsqueeze(0)).square()
        * normalized.unsqueeze(-1)
    ).sum(dim=0)
    pooled_std = pooled_variance.clamp_min(0.0).sqrt()
    support = torch.stack(
        (
            weight.mean(dim=0),
            weight.amax(dim=0),
            weight.gt(1.0e-6).float().mean(dim=0),
            denominator.div(float(weight.shape[0])).clamp(0.0, 1.0),
        ),
        dim=-1,
    )
    return torch.cat((pooled_mean, pooled_std, support), dim=-1).float()


@torch.no_grad()
def extract_direct_occupancy_objects(
    dataset: C1MapTargetDataset,
    *,
    policy: str,
    target_mode: str,
    device: torch.device,
    max_samples: int = 0,
    max_score_diff: float = 1.0e-3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract M0/M1/M2 inputs through the exact frozen C0.3 path."""

    head, checkpoint, amp_dtype = load_frozen_correspondence_head(
        dataset, device=device
    )
    count = len(dataset) if int(max_samples) <= 0 else min(len(dataset), int(max_samples))
    objects: list[dict[str, Any]] = []
    score_max_abs_diff = 0.0
    for index in range(count):
        item = dataset[index]
        uid = str(item["uid"])
        sample = dataset.cache[dataset.cache_by_uid[uid]]
        exact_correct = build_view_identity_evidence(
            sample, device=device, mode="correct"
        )
        exact_weight = exact_correct["view_weight"].float()
        evidence = {
            "correct": exact_correct,
            **{
                control: build_view_identity_evidence(
                    sample, device=device, mode=source
                )
                for control, source in CORRUPTION_POLICY_NAMES.items()
            },
        }
        smoothed: dict[str, dict[str, Any]] = {}
        fixed_weight: torch.Tensor | None = None
        for name, branch in evidence.items():
            smoothed_branch, branch_weight = apply_symmetric_spatial_tolerance(
                branch,
                fixed_correct_weight=exact_weight,
                mode="gaussian3",
            )
            smoothed[name] = smoothed_branch
            if fixed_weight is None:
                fixed_weight = branch_weight
            elif not torch.equal(fixed_weight, branch_weight):
                raise RuntimeError("C1.1b branch support differs after gaussian3")
        assert fixed_weight is not None
        branch_rows: dict[str, Any] = {}
        for name, branch in smoothed.items():
            with _amp_context(device, amp_dtype):
                result = head(branch, view_weight_override=fixed_weight)
                pooled = pooled_view_geometry_features(head, branch, fixed_weight)
            branch_rows[name] = {
                "view_geometry": pooled.float().cpu(),
                "voxel_score": result["voxel_score"].float().cpu(),
            }
        saved_correct = item["map"]["correct_score"].float().reshape(-1)
        current_correct = branch_rows["correct"]["voxel_score"].reshape(-1)
        score_diff = float((saved_correct - current_correct).abs().max().item())
        score_max_abs_diff = max(score_max_abs_diff, score_diff)
        if score_diff > float(max_score_diff):
            raise RuntimeError(
                f"uid={uid} frozen score reconstruction diff={score_diff} "
                f"> {max_score_diff}"
            )
        matched, invariants = matched_candidate_weights(
            item["map"], policy=policy, uid=uid
        )
        active = item["map"]["active_mask"].bool().reshape(-1)
        reliability = item["map"]["audit_maps"]["raw_reliability"].float().reshape(-1)
        candidates = {
            name: {
                "base": torch.cat(
                    (
                        reliability.unsqueeze(-1),
                        branch_rows[name]["view_geometry"],
                    ),
                    dim=-1,
                ).cpu(),
                "correspondence": matched[name].float().reshape(-1, 1).cpu(),
            }
            for name in DIRECT_BRANCHES
        }
        objects.append(
            {
                "uid": uid,
                "object_uid": item["object_uid"],
                "views": int(item["views"]),
                "active": active.cpu(),
                "reliability": reliability.reshape(-1, 1).cpu(),
                "target": item["targets"][target_mode].float().reshape(-1).cpu(),
                "candidates": candidates,
                "target_mapping_audit": item["target_mapping_audit"],
                "matched_budget_invariants": invariants,
            }
        )
        print(
            f"[c1_1b_extract] {index + 1}/{count} uid={uid} "
            f"views={item['views']} score_diff={score_diff:.3e}",
            flush=True,
        )
    metadata = {
        "version": C1_DIRECT_OCCUPANCY_VERSION,
        "source_checkpoint": dataset.report["checkpoint"],
        "source_checkpoint_sha256": dataset.report["checkpoint_sha256"],
        "source_checkpoint_step": int(checkpoint["step"]),
        "source_cache_config_hash": dataset.report["cache_config_hash"],
        "source_c0_report": str(dataset.report_path),
        "policy": policy,
        "target_mode": target_mode,
        "spatial_tolerance": "gaussian3",
        "fixed_correct_support": True,
        "base_feature_definition": (
            "raw reliability + frozen C0 joint per-view token weighted mean/std "
            "+ four fixed-support statistics"
        ),
        "correspondence_feature_definition": (
            "one histogram-matched C1.0b policy weight"
        ),
        "score_reconstruction_max_abs_diff": score_max_abs_diff,
        "object_count": len(objects),
        "input_dim": int(objects[0]["candidates"]["correct"]["base"].shape[-1]),
    }
    del head
    return objects, metadata


def fit_normalization(objects: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    base = torch.cat(
        [row["candidates"]["correct"]["base"][row["active"]] for row in objects],
        dim=0,
    ).float()
    correspondence = torch.cat(
        [
            row["candidates"]["correct"]["correspondence"][row["active"]]
            for row in objects
        ],
        dim=0,
    ).float()
    return {
        "base_mean": base.mean(dim=0),
        "base_std": base.std(dim=0, unbiased=False).clamp_min(1.0e-4),
        "correspondence_mean": correspondence.mean(dim=0),
        "correspondence_std": correspondence.std(dim=0, unbiased=False).clamp_min(1.0e-4),
    }


def normalize_features(
    base: torch.Tensor,
    correspondence: torch.Tensor,
    normalization: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_base = (base.float() - normalization["base_mean"]) / normalization[
        "base_std"
    ]
    normalized_correspondence = (
        correspondence.float() - normalization["correspondence_mean"]
    ) / normalization["correspondence_std"]
    return normalized_base, normalized_correspondence


def model_logits(
    models: dict[str, nn.Module],
    *,
    reliability: torch.Tensor,
    base: torch.Tensor,
    correspondence: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "M0_reliability": models["M0_reliability"](reliability),
        "M1_view_geometry": models["M1_view_geometry"](base),
        "M2_plus_correspondence": models["M2_plus_correspondence"](
            base, correspondence
        ),
    }


def occupancy_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    from pose_point_depth_mv.c1_occupancy import average_precision, roc_auc

    logits = logits.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    if logits.shape != target.shape or logits.numel() == 0:
        raise ValueError("occupancy metric inputs must be non-empty and aligned")
    probability = torch.sigmoid(logits)
    return {
        "balanced_bce": float(balanced_binary_loss(logits, target).item()),
        "average_precision": float(average_precision(probability, target.bool())),
        "roc_auc": float(roc_auc(probability, target.bool())),
        "probability_mean": float(probability.mean().item()),
    }
