#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def parse_fractions(value: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        fractions = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    else:
        fractions = tuple(float(item) for item in value)
    if not fractions:
        raise ValueError("at least one top fraction is required")
    for fraction in fractions:
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"top fraction must be in (0,1), got {fraction}")
    if len(set(fractions)) != len(fractions):
        raise ValueError(f"duplicate top fractions={fractions}")
    return fractions


def stable_top_fraction_mask(
    values: np.ndarray,
    valid: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Select exactly ceil(fraction * finite_valid_count) highest entries.

    Ties are broken deterministically by flat index. The mask is always a
    subset of ``valid`` and is therefore suitable for equal-coverage audits.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if values.shape != valid.shape:
        raise ValueError("values/valid shape mismatch")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must be in (0,1)")
    indices = np.flatnonzero(valid & np.isfinite(values))
    mask = np.zeros(values.shape, dtype=bool)
    if indices.size == 0:
        return mask
    count = max(1, int(np.ceil(float(fraction) * float(indices.size))))
    # np.lexsort uses the last key as primary: descending score, then index.
    order = np.lexsort((indices, -values[indices]))
    mask[indices[order[:count]]] = True
    return mask


def random_fraction_mask(
    valid: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must be in (0,1)")
    indices = np.flatnonzero(valid)
    mask = np.zeros(valid.shape, dtype=bool)
    if indices.size == 0:
        return mask
    count = max(1, int(np.ceil(float(fraction) * float(indices.size))))
    mask[rng.choice(indices, size=count, replace=False)] = True
    return mask


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if values.shape != mask.shape:
        raise ValueError("values/mask shape mismatch")
    selected = values[mask & np.isfinite(values)]
    return float(np.mean(selected)) if selected.size else 0.0


def percentile_rank_gate(
    values: np.ndarray,
    valid: np.ndarray,
    fraction: float,
    *,
    soft: bool,
) -> np.ndarray:
    """Create a per-volume hard top-fraction or rank-ramp soft gate.

    The soft gate is zero below percentile ``1-fraction`` and ramps linearly to
    one at the highest-ranked valid voxel. Because it uses ranks, it is
    invariant to any strictly monotonic confidence rescaling.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if values.shape != valid.shape:
        raise ValueError("values/valid shape mismatch")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must be in (0,1)")
    hard = stable_top_fraction_mask(values, valid, fraction)
    if not soft:
        return hard.astype(np.float32)

    indices = np.flatnonzero(valid & np.isfinite(values))
    gate = np.zeros(values.shape, dtype=np.float32)
    if indices.size == 0:
        return gate
    order = np.lexsort((indices, values[indices]))
    ranks = np.empty(indices.size, dtype=np.float64)
    if indices.size == 1:
        ranks[0] = 1.0
    else:
        ranks[order] = np.linspace(0.0, 1.0, num=indices.size, endpoint=True)
    start = 1.0 - float(fraction)
    local = np.clip((ranks - start) / float(fraction), 0.0, 1.0)
    gate[indices] = local.astype(np.float32)
    return gate


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != weights.shape:
        raise ValueError("values/weights shape mismatch")
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    denominator = float(np.sum(weights[finite]))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(values[finite] * weights[finite]) / denominator)


@dataclass(frozen=True)
class SelectorMeans:
    overall_wrong: float
    correct_selected_wrong: float
    wrong_selected_wrong: float
    shuffle_selected_wrong: float
    random_selected_wrong: float
    overall_shuffle: float
    correct_selected_shuffle: float
    wrong_selected_shuffle: float
    shuffle_selected_shuffle: float
    random_selected_shuffle: float
    selected_count: int
    valid_count: int


_FLOAT_FIELDS = (
    "overall_wrong",
    "correct_selected_wrong",
    "wrong_selected_wrong",
    "shuffle_selected_wrong",
    "random_selected_wrong",
    "overall_shuffle",
    "correct_selected_shuffle",
    "wrong_selected_shuffle",
    "shuffle_selected_shuffle",
    "random_selected_shuffle",
)


def aggregate_selector_means(rows: Sequence[SelectorMeans]) -> SelectorMeans:
    """Average per-volume selector metrics and sum voxel counts.

    This deliberately gives every held-out volume equal weight before the
    caller aggregates to object level. Large-support volumes cannot dominate
    the chosen top percentile merely because they contain more valid voxels.
    """
    if not rows:
        raise ValueError("at least one SelectorMeans row is required")
    values = {
        name: float(np.mean([float(getattr(row, name)) for row in rows]))
        for name in _FLOAT_FIELDS
    }
    return SelectorMeans(
        **values,
        selected_count=int(sum(int(row.selected_count) for row in rows)),
        valid_count=int(sum(int(row.valid_count) for row in rows)),
    )


def evaluate_object_selectors(
    *,
    correct_confidence: np.ndarray,
    wrong_confidence: np.ndarray,
    shuffle_confidence: np.ndarray,
    reprojection_advantage: np.ndarray,
    shuffle_reprojection_advantage: np.ndarray,
    valid: np.ndarray,
    fraction: float,
    random_trials: int,
    seed: int,
) -> SelectorMeans:
    """Evaluate selectors for one held-out voxel volume.

    The four selectors all choose exactly the same number of valid voxels:
    correct-confidence top fraction, wrong-confidence top fraction,
    shuffle-confidence top fraction, and an equal-coverage random baseline.
    """
    arrays = [
        np.asarray(correct_confidence).reshape(-1),
        np.asarray(wrong_confidence).reshape(-1),
        np.asarray(shuffle_confidence).reshape(-1),
        np.asarray(reprojection_advantage).reshape(-1),
        np.asarray(shuffle_reprojection_advantage).reshape(-1),
        np.asarray(valid, dtype=bool).reshape(-1),
    ]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("all selector arrays must have the same shape")
    c, w, s, adv_w, adv_s, valid_mask = arrays
    finite = (
        valid_mask
        & np.isfinite(c)
        & np.isfinite(w)
        & np.isfinite(s)
        & np.isfinite(adv_w)
        & np.isfinite(adv_s)
    )
    if not np.any(finite):
        raise ValueError("no finite valid voxels")

    c_mask = stable_top_fraction_mask(c, finite, fraction)
    w_mask = stable_top_fraction_mask(w, finite, fraction)
    s_mask = stable_top_fraction_mask(s, finite, fraction)
    rng = np.random.default_rng(int(seed))
    random_wrong: list[float] = []
    random_shuffle: list[float] = []
    for _ in range(max(1, int(random_trials))):
        random_mask = random_fraction_mask(finite, fraction, rng)
        random_wrong.append(masked_mean(adv_w, random_mask))
        random_shuffle.append(masked_mean(adv_s, random_mask))

    return SelectorMeans(
        overall_wrong=masked_mean(adv_w, finite),
        correct_selected_wrong=masked_mean(adv_w, c_mask),
        wrong_selected_wrong=masked_mean(adv_w, w_mask),
        shuffle_selected_wrong=masked_mean(adv_w, s_mask),
        random_selected_wrong=float(np.mean(random_wrong)),
        overall_shuffle=masked_mean(adv_s, finite),
        correct_selected_shuffle=masked_mean(adv_s, c_mask),
        wrong_selected_shuffle=masked_mean(adv_s, w_mask),
        shuffle_selected_shuffle=masked_mean(adv_s, s_mask),
        random_selected_shuffle=float(np.mean(random_shuffle)),
        selected_count=int(c_mask.sum()),
        valid_count=int(finite.sum()),
    )
