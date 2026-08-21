from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean, median
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv.correspondence_head import (
    continuous_voxel_gate_weight,
    hard_admitted_soft_weight,
)


C1_OCCUPANCY_PROTOCOL_VERSION = "pose_point_depth_mv.c1_occupancy.v2"
C1_ENRICHMENT_REPORT_VERSION = "pose_point_depth_mv.c1_enrichment_report.v2"
C1_ENRICHMENT_SUMMARY_VERSION = "pose_point_depth_mv.c1_enrichment_summary.v2"
C1_CALIBRATOR_VERSION = "pose_point_depth_mv.c1_nested_monotone_calibrator.v2"
C1_CALIBRATOR_CHECKPOINT_VERSION = (
    "pose_point_depth_mv.c1_nested_monotone_calibrator_checkpoint.v2"
)

PRIMARY_POLICIES = (
    "hard_binary",
    "hard_admitted",
    "continuous",
)
BASELINE_POLICIES = (
    "active_only",
    "reliability_only",
)
TARGET_MODES = ("exact", "surface_r1")
SEMANTIC_NAMES = {
    1: "surface",
    2: "free_space",
    3: "occluded",
    4: "boundary",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    text = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def target_occupancy_masks(target_coords: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map the existing 64^3 sparse target coordinates to the SS 16^3 grid."""

    coords = target_coords.detach().long()[..., -3:]
    valid = ((coords >= 0) & (coords < 64)).all(dim=1)
    coords16 = torch.div(coords[valid], 4, rounding_mode="floor").clamp(0, 15)
    exact = torch.zeros((1, 1, 16, 16, 16), dtype=torch.float32)
    if coords16.numel():
        exact[0, 0, coords16[:, 0], coords16[:, 1], coords16[:, 2]] = 1.0
    surface_r1 = F.max_pool3d(exact, kernel_size=3, stride=1, padding=1)
    return {
        "exact": exact[0, 0].bool(),
        "surface_r1": surface_r1[0, 0].bool(),
    }


def mask_sha256(mask: torch.Tensor) -> str:
    values = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def target_mapping_audit(target_coords: torch.Tensor) -> dict[str, Any]:
    """Audit target [x,y,z] indices against the C0 canonical token ordering."""

    # Import the exact helper used to construct C0 view-identity evidence. This
    # makes the sentinel test fail if that implementation's axis order changes.
    from pose_point_depth_mv.view_identity_lifting import _canonical_xyz

    coords64 = target_coords.detach().to(device="cpu", dtype=torch.long)[..., -3:]
    valid = ((coords64 >= 0) & (coords64 < 64)).all(dim=1)
    coords64 = coords64[valid]
    if coords64.numel() == 0:
        raise ValueError("target mapping audit requires at least one valid coordinate")
    coords16 = torch.div(coords64, 4, rounding_mode="floor").clamp(0, 15)
    unique16 = torch.unique(coords16, dim=0)
    distinct = (unique16[:, 0] != unique16[:, 1]) & (
        unique16[:, 1] != unique16[:, 2]
    )
    sentinel = unique16[torch.nonzero(distinct, as_tuple=False)[0, 0]] if bool(
        distinct.any().item()
    ) else unique16[0]

    side = 16
    canonical = _canonical_xyz(
        side, device=torch.device("cpu"), dtype=torch.float32
    ).reshape(side, side, side, 3)
    canonical_center = canonical[
        int(sentinel[0]), int(sentinel[1]), int(sentinel[2])
    ]
    expected_center = (sentinel.float() + 0.5) / float(side) * 2.0 - 1.0
    inverse = torch.floor(
        (canonical_center + 1.0) * 0.5 * float(side)
    ).long().clamp(0, side - 1)
    flat_formula = int(
        sentinel[0] * side * side + sentinel[1] * side + sentinel[2]
    )
    flat_grid = torch.arange(side**3).reshape(side, side, side)
    flat_lookup = int(
        flat_grid[int(sentinel[0]), int(sentinel[1]), int(sentinel[2])].item()
    )

    matching64 = coords64[(coords16 == sentinel).all(dim=1)][0]
    target_center = (matching64.float() + 0.5) / 64.0 * 2.0 - 1.0
    target_inverse = torch.floor(
        (target_center + 1.0) * 0.5 * float(side)
    ).long().clamp(0, side - 1)
    masks = target_occupancy_masks(coords64)
    checks = {
        "valid_target_coordinates_present": bool(coords64.numel()),
        "canonical_center_matches_xyz_index": bool(
            torch.equal(canonical_center, expected_center)
        ),
        "canonical_center_roundtrip": bool(torch.equal(inverse, sentinel)),
        "target_center_roundtrip": bool(torch.equal(target_inverse, sentinel)),
        "flat_index_matches_xyz_formula": flat_lookup == flat_formula,
        "sentinel_marks_exact_target_mask": bool(
            masks["exact"][int(sentinel[0]), int(sentinel[1]), int(sentinel[2])]
        ),
    }
    return {
        "axis_order": "[x,y,z]",
        "canonical_source": (
            "pose_point_depth_mv.view_identity_lifting._canonical_xyz"
        ),
        "flatten_formula": "x*16*16+y*16+z",
        "sentinel_coord64": [int(value) for value in matching64.tolist()],
        "sentinel_coord16": [int(value) for value in sentinel.tolist()],
        "sentinel_has_distinct_axes": bool(distinct.any().item()),
        "canonical_center": [float(value) for value in canonical_center.tolist()],
        "flat_index": flat_lookup,
        "exact_target_mask_sha256": mask_sha256(masks["exact"]),
        "surface_r1_mask_sha256": mask_sha256(masks["surface_r1"]),
        "checks": checks,
        "passed": all(checks.values()),
    }


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(finite),
        "mean": mean(finite),
        "median": median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def bootstrap_ci(
    values: Iterable[float], *, seed: int = 20260719, samples: int = 10000
) -> list[float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return [0.0, 0.0]
    rng = random.Random(int(seed))
    draws = [
        mean(finite[rng.randrange(len(finite))] for _ in finite)
        for _ in range(int(samples))
    ]
    draws.sort()
    return [
        draws[min(len(draws) - 1, int(0.025 * len(draws)))],
        draws[min(len(draws) - 1, int(0.975 * len(draws)))],
    ]


def positive_rate(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(mean(value > 0.0 for value in finite)) if finite else 0.0


def comparison_summary(
    values: Iterable[float], *, bootstrap_samples: int
) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "object": summarize(finite),
        "object_win_rate": positive_rate(finite),
        "object_bootstrap_95_ci": bootstrap_ci(
            finite, samples=int(bootstrap_samples)
        ),
    }


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Threshold-grouped AP; tied scores are consumed as one PR step."""

    scores = scores.detach().float().reshape(-1)
    labels = labels.detach().bool().reshape(-1)
    positives = int(labels.sum().item())
    if positives == 0:
        return 0.0
    order = torch.argsort(scores, descending=True, stable=True)
    ranked_scores = scores[order]
    ranked_labels = labels[order].float()
    group_end = torch.ones_like(ranked_scores, dtype=torch.bool)
    group_end[:-1] = ranked_scores[:-1].ne(ranked_scores[1:])
    cumulative_tp = ranked_labels.cumsum(0)[group_end]
    cumulative_count = torch.arange(
        1,
        ranked_scores.numel() + 1,
        device=ranked_scores.device,
        dtype=torch.float32,
    )[group_end]
    recall = cumulative_tp / float(positives)
    previous_recall = torch.cat((recall.new_zeros(1), recall[:-1]))
    precision = cumulative_tp / cumulative_count
    return float(((recall - previous_recall) * precision).sum().item())


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Tie-aware Mann-Whitney AUROC without a sklearn dependency."""

    scores = scores.detach().float().reshape(-1)
    labels = labels.detach().bool().reshape(-1)
    positive_count = int(labels.sum().item())
    negative_count = int((~labels).sum().item())
    if positive_count == 0 or negative_count == 0:
        return 0.5
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.empty_like(sorted_scores)
    start = 0
    while start < sorted_scores.numel():
        end = start + 1
        while end < sorted_scores.numel() and bool(
            sorted_scores[end] == sorted_scores[start]
        ):
            end += 1
        # Statistical ranks are one-based.
        ranks[start:end] = 0.5 * ((start + 1) + end)
        start = end
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    original_ranks = ranks[inverse]
    positive_rank_sum = original_ranks[labels].sum()
    numerator = positive_rank_sum - positive_count * (positive_count + 1) / 2
    return float((numerator / (positive_count * negative_count)).item())


def policy_metrics(
    score: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
) -> dict[str, float | int]:
    score = score.detach().float().reshape(-1)
    target = target.detach().bool().reshape(-1)
    active = active_mask.detach().bool().reshape(-1)
    if score.shape != target.shape or score.shape != active.shape:
        raise ValueError("C1 score, target, and active mask shapes must match")
    if not bool(torch.isfinite(score).all().item()):
        raise ValueError("C1 score contains non-finite values")
    if bool((score < 0.0).any().item()):
        raise ValueError("C1 score must be non-negative")
    active_count = int(active.sum().item())
    if active_count == 0:
        raise ValueError("C1 sample has no active voxels")
    active_score = score[active]
    active_target = target[active]
    target_count = int(target.sum().item())
    active_target_count = int(active_target.sum().item())
    score_mass = active_score.sum()
    weighted_target_rate = (
        float((active_score * active_target.float()).sum().div(score_mass).item())
        if float(score_mass.item()) > 0.0
        else 0.0
    )
    support = active_score > 0.0
    support_count = int(support.sum().item())
    support_true = int((support & active_target).sum().item())
    result: dict[str, float | int] = {
        "active_count": active_count,
        "target_count": target_count,
        "active_target_count": active_target_count,
        "active_target_rate": float(active_target.float().mean().item()),
        "score_mass": float(score_mass.item()),
        "weighted_target_rate": weighted_target_rate,
        "weighted_target_coverage": (
            float(
                (active_score * active_target.float())
                .sum()
                .div(max(target_count, 1))
                .item()
            )
        ),
        "support_count": support_count,
        "support_precision": float(support_true / max(support_count, 1)),
        "support_recall": float(support_true / max(target_count, 1)),
        "average_precision_active": average_precision(active_score, active_target),
        "roc_auc_active": roc_auc(active_score, active_target),
    }
    for fraction in (0.05, 0.10, 0.20):
        count = max(1, int(math.ceil(active_count * fraction)))
        selected = torch.topk(active_score, k=min(count, active_count)).indices
        precision = float(active_target[selected].float().mean().item())
        key = f"top_{int(fraction * 100):02d}_target_rate"
        result[key] = precision
        result[f"top_{int(fraction * 100):02d}_enrichment"] = float(
            precision / max(float(result["active_target_rate"]), 1.0e-12)
        )
    return result


def permute_within_active(
    score: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    uid: str,
    repeat: int,
) -> torch.Tensor:
    score = score.detach().float()
    active = active_mask.detach().bool()
    if score.shape != active.shape:
        raise ValueError("permutation score/active shape mismatch")
    indices = torch.nonzero(active.reshape(-1), as_tuple=False).flatten()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(C1_OCCUPANCY_PROTOCOL_VERSION, uid, repeat))
    permutation = torch.randperm(indices.numel(), generator=generator)
    output = torch.zeros_like(score).reshape(-1)
    flat = score.reshape(-1)
    output[indices] = flat[indices][permutation]
    return output.reshape_as(score)


def c1_policy_scores(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    active = payload["active_mask"].bool()
    reliability = payload["audit_maps"]["raw_reliability"].float()
    scores = {
        "active_only": active.float(),
        "reliability_only": reliability.masked_fill(~active, 0.0),
        "hard_binary": payload["gate_mask"].float(),
        "hard_admitted": payload["hard_admitted_soft_weight"].float(),
        "continuous": payload["continuous_soft_weight"].float(),
    }
    protocol = payload["hard_admitted_soft_weight_protocol"]
    correct_score = payload["correct_score"].float()
    training_branch_scores = {
        name: correct_score - margin.float()
        for name, margin in payload.get("training_control_margins", {}).items()
    }
    branch_bank = {"correct": correct_score, **training_branch_scores}
    if not training_branch_scores:
        raise ValueError("C1 map has no reconstructed corruption branch scores")

    def add_corruption_weights(
        name: str, candidate_margin: torch.Tensor, *, prefix: str
    ) -> None:
        scores[f"{prefix}_hard_binary_{name}"] = (
            active & candidate_margin.gt(0.0)
        ).float()
        scores[f"{prefix}_hard_admitted_{name}"] = hard_admitted_soft_weight(
            candidate_margin.reshape(-1),
            reliability.reshape(-1),
            active.reshape(-1),
            temperature=float(protocol["temperature"]),
            reliability_power=float(protocol["reliability_power"]),
        ).reshape_as(reliability)
        continuous_protocol = payload["continuous_soft_weight_protocol"]
        scores[f"{prefix}_continuous_{name}"] = continuous_voxel_gate_weight(
            candidate_margin.reshape(-1),
            reliability.reshape(-1),
            active.reshape(-1),
            temperature=float(continuous_protocol["temperature"]),
            reliability_power=float(continuous_protocol["reliability_power"]),
            max_scale=float(continuous_protocol["max_scale"]),
        ).reshape_as(reliability)

    for name, candidate_score in training_branch_scores.items():
        competitors = torch.stack(
            [score for branch, score in branch_bank.items() if branch != name], dim=0
        )
        candidate_margin = candidate_score - competitors.max(dim=0).values
        add_corruption_weights(name, candidate_margin, prefix="corruption")
    for name, margin in payload.get("heldout_margins", {}).items():
        candidate_score = correct_score - margin.float()
        competitors = torch.stack(list(branch_bank.values()), dim=0)
        candidate_margin = candidate_score - competitors.max(dim=0).values
        add_corruption_weights(name, candidate_margin, prefix="heldout_corruption")
    for name, score in scores.items():
        if score.shape != (16, 16, 16):
            raise ValueError(f"C1 policy {name} has shape {tuple(score.shape)}")
        if not bool(torch.isfinite(score).all().item()):
            raise ValueError(f"C1 policy {name} is non-finite")
        if bool(score[~active].ne(0.0).any().item()):
            raise ValueError(f"C1 policy {name} escaped fixed active support")
    return scores


class C1MapTargetDataset:
    """Join immutable C0 maps with target labels by UID, never by row order."""

    def __init__(self, c0_report: str | Path) -> None:
        self.report_path = Path(c0_report).resolve()
        self.report = load_json(self.report_path)
        if self.report.get("passed") is not True:
            raise ValueError("C1 requires a passed C0.3 report")
        if self.report.get("evaluation_spatial_tolerance") != "gaussian3":
            raise ValueError("C1 requires the admitted gaussian3 C0 protocol")
        self.cache = PoseLiftingCacheDataset(
            self.report["cache_manifest"], indices="all"
        )
        if self.cache.config_hash != self.report.get("cache_config_hash"):
            raise ValueError("C0 report/cache config hash mismatch")
        self.cache_by_uid = {
            str(row["uid"]): index for index, row in enumerate(self.cache.rows)
        }
        records = list(self.report.get("records", []))
        if not records:
            raise ValueError("C0 report has no records")
        self.records = records
        report_uids = [str(row["uid"]) for row in records]
        if len(set(report_uids)) != len(report_uids):
            raise ValueError("C0 report has duplicate UIDs")
        missing = sorted(set(report_uids) - set(self.cache_by_uid))
        if missing:
            raise ValueError(f"C0 UIDs missing from cache: {missing[:3]}")
        self.maps_dir = self.report_path.parent / "voxel_maps"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        uid = str(record["uid"])
        map_path = self.maps_dir / f"{uid}.pt"
        payload = torch.load(map_path, map_location="cpu")
        if str(payload.get("uid")) != uid:
            raise ValueError(f"C1 map UID mismatch for {uid}")
        if str(payload.get("cache_config_hash")) != self.cache.config_hash:
            raise ValueError(f"C1 map cache hash mismatch for {uid}")
        if int(payload.get("checkpoint_step", -1)) != int(
            self.report["checkpoint_step"]
        ):
            raise ValueError(f"C1 map checkpoint step mismatch for {uid}")
        forbidden = {"target", "target_coords", "target_mask"} & set(payload)
        if forbidden:
            raise ValueError(f"C1 map illegally contains target fields: {forbidden}")
        sample = self.cache[self.cache_by_uid[uid]]
        if str(sample["uid"]) != uid:
            raise ValueError(f"C1 cache UID mismatch for {uid}")
        return {
            "uid": uid,
            "object_uid": str(record["object_uid"]),
            "views": int(record["views"]),
            "map_path": str(map_path.resolve()),
            "map": payload,
            "targets": target_occupancy_masks(sample["target_coords"]),
            "target_mapping_audit": target_mapping_audit(sample["target_coords"]),
        }


class MonotoneOccupancyCalibrator(nn.Module):
    """A nested monotone probe with no spatial or Flow features."""

    def __init__(
        self,
        *,
        include_score: bool = True,
        include_reliability: bool = True,
    ) -> None:
        super().__init__()
        self.include_score = bool(include_score)
        self.include_reliability = bool(include_reliability)
        self.bias = nn.Parameter(torch.tensor(0.0))
        if self.include_score:
            self.score_weight_raw = nn.Parameter(torch.tensor(-2.0))
        else:
            self.register_parameter("score_weight_raw", None)
        if self.include_reliability:
            self.reliability_weight_raw = nn.Parameter(torch.tensor(-2.0))
        else:
            self.register_parameter("reliability_weight_raw", None)

    def forward(
        self, score: torch.Tensor, reliability: torch.Tensor
    ) -> torch.Tensor:
        if score.shape != reliability.shape:
            raise ValueError("calibrator score/reliability shapes must match")
        output = self.bias.expand_as(score.float())
        if self.include_score:
            output = output + F.softplus(self.score_weight_raw) * score.float()
        if self.include_reliability:
            output = output + F.softplus(self.reliability_weight_raw) * reliability.float()
        return output

    def metadata(self) -> dict[str, Any]:
        return {
            "version": C1_CALIBRATOR_VERSION,
            "type": type(self).__name__,
            "include_score": self.include_score,
            "include_reliability": self.include_reliability,
            "uses_xyz": False,
            "uses_flow_state": False,
            "uses_convolution": False,
            "score_weight_present": self.include_score,
            "score_weight_nonnegative": True if self.include_score else None,
            "reliability_weight_present": self.include_reliability,
            "reliability_weight_nonnegative": (
                True if self.include_reliability else None
            ),
        }


def balanced_binary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = logits.float().reshape(-1)
    target = target.float().reshape(-1)
    positive = target > 0.5
    negative = ~positive
    if not bool(positive.any().item()) or not bool(negative.any().item()):
        raise ValueError("balanced C1 loss requires positive and negative voxels")
    positive_loss = F.binary_cross_entropy_with_logits(
        logits[positive], target[positive]
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        logits[negative], target[negative]
    )
    return 0.5 * (positive_loss + negative_loss)


def calibrator_parameter_values(model: MonotoneOccupancyCalibrator) -> dict[str, float]:
    values = {"bias": float(model.bias.detach().item())}
    if model.score_weight_raw is not None:
        values["score_weight"] = float(
            F.softplus(model.score_weight_raw.detach()).item()
        )
    if model.reliability_weight_raw is not None:
        values["reliability_weight"] = float(
            F.softplus(model.reliability_weight_raw.detach()).item()
        )
    return values


def protocol_signature(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": report.get("format"),
        "target_mapping": report.get("target_protocol"),
        "policy_protocol": report.get("policy_protocol"),
        "decision_thresholds": report.get("decision_thresholds"),
        "permutation_repeats": report.get("permutation_repeats"),
    }


def group_records(
    records: list[dict[str, Any]], key: str
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row[key])].append(row)
    return grouped
