from __future__ import annotations

import hashlib
import math
from typing import Any

import torch

from pose_point_depth_mv.c1_occupancy import (
    SEMANTIC_NAMES,
    c1_policy_scores,
    stable_seed,
)


C1_MATCHED_BUDGET_VERSION = "pose_point_depth_mv.c1_matched_budget.v1"
C1_MATCHED_BUDGET_REPORT_VERSION = (
    "pose_point_depth_mv.c1_matched_budget_report.v1"
)
C1_MATCHED_BUDGET_SUMMARY_VERSION = (
    "pose_point_depth_mv.c1_matched_budget_summary.v1"
)

MATCHED_POLICIES = ("hard_admitted", "continuous")
MATCHED_CONTROLS = (
    "reliability",
    "pose_corruption",
    "depth_corruption",
    "visual_corruption",
    "spatial_permutation",
)
CORRUPTION_POLICY_NAMES = {
    "pose_corruption": "pose_cyclic1",
    "depth_corruption": "depth_view_cyclic1",
    "visual_corruption": "visual_view_cyclic1",
}
DEFAULT_BUDGET_FRACTIONS = (0.05, 0.10, 0.20)


def parse_budget_fractions(value: str) -> tuple[float, ...]:
    fractions = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not fractions or len(set(fractions)) != len(fractions):
        raise ValueError("budget fractions must be non-empty and unique")
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError("budget fractions must be in (0,1]")
    return tuple(sorted(fractions))


def budget_key(fraction: float) -> str:
    return f"top_{int(round(100.0 * float(fraction))):02d}"


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def deterministic_rank_order(
    score: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    uid: str,
    name: str,
) -> torch.Tensor:
    """Rank active voxels with deterministic, target-independent tie breaking."""

    values = score.detach().float().reshape(-1)
    active = active_mask.detach().bool().reshape(-1)
    if values.shape != active.shape:
        raise ValueError("rank score and active mask shapes differ")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("rank score is non-finite")
    indices = torch.nonzero(active, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError("cannot rank an empty active support")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(C1_MATCHED_BUDGET_VERSION, uid, name))
    tie_order = torch.randperm(indices.numel(), generator=generator)
    shuffled = indices[tie_order]
    primary = torch.argsort(values[shuffled], descending=True, stable=True)
    return shuffled[primary]


def histogram_match_to_ranking(
    reference_weight: torch.Tensor,
    ranking_score: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    uid: str,
    name: str,
) -> torch.Tensor:
    """Assign the exact reference histogram according to another voxel ranking."""

    reference = reference_weight.detach().float().reshape(-1)
    ranking = ranking_score.detach().float().reshape(-1)
    active = active_mask.detach().bool().reshape(-1)
    if reference.shape != ranking.shape or reference.shape != active.shape:
        raise ValueError("histogram matching shapes differ")
    if bool(reference[~active].ne(0.0).any().item()):
        raise ValueError("reference weight escaped active support")
    if bool((reference < 0.0).any().item()) or not bool(
        torch.isfinite(reference).all().item()
    ):
        raise ValueError("reference weights must be finite and non-negative")
    order = deterministic_rank_order(ranking, active, uid=uid, name=name)
    sorted_weight = torch.sort(reference[active], descending=True).values
    output = torch.zeros_like(reference)
    output[order] = sorted_weight
    return output.reshape_as(reference_weight)


def spatially_permuted_score(
    score: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    uid: str,
) -> torch.Tensor:
    """Deterministically break the score-to-voxel binding within fixed support."""

    values = score.detach().float().reshape(-1)
    active = active_mask.detach().bool().reshape(-1)
    indices = torch.nonzero(active, as_tuple=False).flatten()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        stable_seed(C1_MATCHED_BUDGET_VERSION, uid, "spatial_permutation")
    )
    permutation = torch.randperm(indices.numel(), generator=generator)
    output = torch.zeros_like(values)
    output[indices] = values[indices][permutation]
    return output.reshape_as(score)


def matched_candidate_weights(
    payload: dict[str, Any],
    *,
    policy: str,
    uid: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build correct and controls with identical support and weight histogram."""

    if policy not in MATCHED_POLICIES:
        raise ValueError(f"unsupported matched-budget policy={policy!r}")
    active = payload["active_mask"].bool()
    policy_scores = c1_policy_scores(payload)
    reference = policy_scores[policy].float()
    raw_reliability = payload["audit_maps"]["raw_reliability"].float()
    ranking_scores: dict[str, torch.Tensor] = {
        "correct": reference,
        "reliability": raw_reliability,
        "spatial_permutation": spatially_permuted_score(
            reference, active, uid=uid
        ),
    }
    for control, source in CORRUPTION_POLICY_NAMES.items():
        key = f"corruption_{policy}_{source}"
        if key not in policy_scores:
            raise ValueError(f"missing real corruption ranking {key}")
        ranking_scores[control] = policy_scores[key].float()

    matched = {
        name: histogram_match_to_ranking(
            reference,
            ranking,
            active,
            uid=uid,
            name=f"{policy}:{name}",
        )
        for name, ranking in ranking_scores.items()
    }
    reference_histogram = torch.sort(reference[active], descending=True).values
    invariants: dict[str, Any] = {
        "active_count": int(active.sum().item()),
        "reference_nonzero_count": int(reference[active].gt(0.0).sum().item()),
        "reference_mass": float(reference[active].sum().item()),
        "reference_histogram_sha256": tensor_sha256(reference_histogram),
        "candidates": {},
    }
    for name, weight in matched.items():
        candidate_histogram = torch.sort(weight[active], descending=True).values
        invariants["candidates"][name] = {
            "histogram_equal": bool(torch.equal(candidate_histogram, reference_histogram)),
            "support_equal": bool(
                torch.equal(weight.ne(0.0), reference.ne(0.0))
                if name == "correct"
                else int(weight[active].gt(0.0).sum().item())
                == int(reference[active].gt(0.0).sum().item())
            ),
            "inactive_zero": bool(weight[~active].eq(0.0).all().item()),
            "mass_abs_diff": float(
                (weight[active].sum() - reference[active].sum()).abs().item()
            ),
            "histogram_sha256": tensor_sha256(candidate_histogram),
        }
    return matched, invariants


def matched_budget_metrics(
    weight: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
    semantic_label: torch.Tensor,
    *,
    uid: str,
    name: str,
    fractions: tuple[float, ...],
) -> dict[str, Any]:
    """Measure a fixed-histogram candidate at fixed top-K budgets."""

    values = weight.detach().float().reshape(-1)
    truth = target.detach().bool().reshape(-1)
    active = active_mask.detach().bool().reshape(-1)
    semantics = semantic_label.detach().long().reshape(-1)
    if not (values.shape == truth.shape == active.shape == semantics.shape):
        raise ValueError("matched-budget metric shapes differ")
    active_count = int(active.sum().item())
    target_count = int(truth.sum().item())
    active_target_count = int((truth & active).sum().item())
    mass = values[active].sum()
    weighted_true_mass = (values * truth.float())[active].sum()
    order = deterministic_rank_order(values, active, uid=uid, name=f"metric:{name}")
    result: dict[str, Any] = {
        "active_count": active_count,
        "target_count": target_count,
        "active_target_count": active_target_count,
        "score_mass": float(mass.item()),
        "nonzero_count": int(values[active].gt(0.0).sum().item()),
        "weighted_target_rate": (
            float(weighted_true_mass.div(mass).item())
            if float(mass.item()) > 0.0
            else 0.0
        ),
        "weighted_target_coverage": float(
            weighted_true_mass.div(max(target_count, 1)).item()
        ),
        "budgets": {},
        "semantics": {},
    }
    for fraction in fractions:
        count = min(active_count, max(1, int(math.ceil(active_count * fraction))))
        selected = order[:count]
        true_count = int(truth[selected].sum().item())
        result["budgets"][budget_key(fraction)] = {
            "fraction": float(fraction),
            "count": count,
            "target_count": true_count,
            "target_rate": float(true_count / count),
            "target_coverage": float(true_count / max(target_count, 1)),
        }
    for label_id, semantic_name in SEMANTIC_NAMES.items():
        mask = active & semantics.eq(int(label_id))
        semantic_mass = values[mask].sum()
        semantic_true_mass = (values * truth.float())[mask].sum()
        result["semantics"][semantic_name] = {
            "voxel_count": int(mask.sum().item()),
            "score_mass": float(semantic_mass.item()),
            "score_mass_fraction": float(
                semantic_mass.div(mass.clamp_min(1.0e-12)).item()
            ),
            "target_rate": (
                float(truth[mask].float().mean().item())
                if bool(mask.any().item())
                else 0.0
            ),
            "weighted_target_rate": (
                float(semantic_true_mass.div(semantic_mass).item())
                if float(semantic_mass.item()) > 0.0
                else 0.0
            ),
        }
    return result

