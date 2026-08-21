#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def parse_int_csv(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        items = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    else:
        items = tuple(int(x) for x in value)
    if not items:
        raise ValueError("at least one integer is required")
    if any(x <= 0 for x in items):
        raise ValueError(f"all values must be positive, got {items}")
    if len(set(items)) != len(items):
        raise ValueError(f"duplicate values={items}")
    return items


def trimmed_mean(values: np.ndarray, trim_fraction: float = 0.10) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if not 0.0 <= float(trim_fraction) < 0.5:
        raise ValueError("trim_fraction must be in [0,0.5)")
    x = np.sort(x)
    trim = int(np.floor(float(trim_fraction) * x.size))
    if trim > 0 and 2 * trim < x.size:
        x = x[trim:-trim]
    return float(np.mean(x))


def region_index(volume_side: int, divisions: int) -> np.ndarray:
    side = int(volume_side)
    div = int(divisions)
    if side <= 0 or div <= 0 or side % div != 0:
        raise ValueError(f"volume_side={side} must be divisible by divisions={div}")
    block = side // div
    x, y, z = np.meshgrid(
        np.arange(side), np.arange(side), np.arange(side), indexing="ij"
    )
    rid = (x // block) * (div * div) + (y // block) * div + (z // block)
    return rid.reshape(-1).astype(np.int32)


@dataclass(frozen=True)
class RegionValues:
    scores: np.ndarray
    counts: np.ndarray
    valid: np.ndarray
    object_score: float


def reduce_regions(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    volume_side: int,
    divisions: int,
    trim_fraction: float,
    min_region_voxels: int,
) -> RegionValues:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if x.shape != mask.shape:
        raise ValueError("values/valid shape mismatch")
    expected = int(volume_side) ** 3
    if x.size != expected:
        raise ValueError(f"expected {expected} voxels, got {x.size}")
    finite = mask & np.isfinite(x)
    object_score = trimmed_mean(x[finite], trim_fraction)
    ids = region_index(volume_side, divisions)
    region_count = int(divisions) ** 3
    scores = np.full(region_count, np.nan, dtype=np.float64)
    counts = np.zeros(region_count, dtype=np.int32)
    for rid in range(region_count):
        local = finite & (ids == rid)
        counts[rid] = int(local.sum())
        if counts[rid] >= int(min_region_voxels):
            scores[rid] = trimmed_mean(x[local], trim_fraction)
    valid_regions = np.isfinite(scores)
    return RegionValues(scores=scores, counts=counts, valid=valid_regions, object_score=object_score)


def shrink_region_scores(
    region: RegionValues,
    *,
    kappa: float | None,
) -> RegionValues:
    if kappa is None:
        return region
    if float(kappa) <= 0.0:
        raise ValueError("kappa must be positive")
    scores = region.scores.copy()
    alpha = region.counts.astype(np.float64) / (
        region.counts.astype(np.float64) + float(kappa)
    )
    valid = region.valid & np.isfinite(region.object_score)
    scores[valid] = (
        alpha[valid] * scores[valid]
        + (1.0 - alpha[valid]) * float(region.object_score)
    )
    return RegionValues(
        scores=scores,
        counts=region.counts.copy(),
        valid=valid,
        object_score=region.object_score,
    )



def bounded_region_gate(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Use the aggregated visual-only confidence directly as a [0,1] gate."""
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid, dtype=bool).reshape(-1) & np.isfinite(x)
    if x.shape != mask.shape:
        raise ValueError("scores/valid shape mismatch")
    gate = np.zeros(x.shape, dtype=np.float64)
    gate[mask] = np.clip(x[mask], 0.0, 1.0)
    return gate

def percentile_region_gate(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return deterministic [0,1] region-rank gates.

    The lowest valid region receives 1/K and the highest receives 1. Ties are
    averaged. A single valid region receives 1, which is the object baseline.
    """
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid, dtype=bool).reshape(-1) & np.isfinite(x)
    if x.shape != mask.shape:
        raise ValueError("scores/valid shape mismatch")
    gate = np.zeros(x.shape, dtype=np.float64)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return gate
    if idx.size == 1:
        gate[idx[0]] = 1.0
        return gate
    order = idx[np.argsort(x[idx], kind="mergesort")]
    cursor = 0
    while cursor < order.size:
        end = cursor + 1
        while end < order.size and x[order[end]] == x[order[cursor]]:
            end += 1
        average_rank = 0.5 * ((cursor + 1) + end)
        gate[order[cursor:end]] = average_rank / float(order.size)
        cursor = end
    return gate


def weighted_region_mean(
    values: np.ndarray,
    gate: np.ndarray,
    counts: np.ndarray,
    valid: np.ndarray,
) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    g = np.asarray(gate, dtype=np.float64).reshape(-1)
    n = np.asarray(counts, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if len({x.shape, g.shape, n.shape, mask.shape}) != 1:
        raise ValueError("region arrays have mismatched shapes")
    finite = mask & np.isfinite(x) & np.isfinite(g) & (g >= 0.0) & (n > 0.0)
    weight = g[finite] * n[finite]
    denominator = float(np.sum(weight))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(x[finite] * weight) / denominator)


def region_label_means(
    labels: np.ndarray,
    valid_voxels: np.ndarray,
    *,
    volume_side: int,
    divisions: int,
    min_region_voxels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid_voxels, dtype=bool).reshape(-1)
    if y.shape != mask.shape:
        raise ValueError("labels/valid shape mismatch")
    ids = region_index(volume_side, divisions)
    count = int(divisions) ** 3
    means = np.full(count, np.nan, dtype=np.float64)
    counts = np.zeros(count, dtype=np.int32)
    finite = mask & np.isfinite(y)
    for rid in range(count):
        local = finite & (ids == rid)
        counts[rid] = int(local.sum())
        if counts[rid] >= int(min_region_voxels):
            means[rid] = float(np.mean(y[local]))
    valid_regions = np.isfinite(means)
    return means, counts, valid_regions


def rank_correlation(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid, dtype=bool).reshape(-1) & np.isfinite(a) & np.isfinite(b)
    idx = np.flatnonzero(mask)
    if idx.size < 3:
        return 0.0

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        out = np.empty(v.size, dtype=np.float64)
        cursor = 0
        while cursor < v.size:
            end = cursor + 1
            while end < v.size and v[order[end]] == v[order[cursor]]:
                end += 1
            out[order[cursor:end]] = 0.5 * ((cursor + 1) + end)
            cursor = end
        return out

    ra = ranks(a[idx])
    rb = ranks(b[idx])
    ra -= np.mean(ra)
    rb -= np.mean(rb)
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    return float(np.sum(ra * rb) / denom) if denom > 1.0e-12 else 0.0


def binary_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative, dtype=np.float64).reshape(-1)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size == 0 or negative.size == 0:
        return 0.5
    values = np.concatenate([positive, negative])
    labels = np.concatenate(
        [np.ones(positive.size, dtype=np.int8), np.zeros(negative.size, dtype=np.int8)]
    )
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and sorted_values[end] == sorted_values[cursor]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * ((cursor + 1) + end)
        cursor = end
    rank_sum = float(np.sum(ranks[labels == 1]))
    return float(
        (rank_sum - positive.size * (positive.size + 1) / 2.0)
        / (positive.size * negative.size)
    )


@dataclass(frozen=True)
class Candidate:
    name: str
    divisions: int
    shrinkage_kappa: float | None


def build_candidates(
    divisions: Iterable[int], shrinkage_kappas: Iterable[int]
) -> tuple[Candidate, ...]:
    result: list[Candidate] = []
    for div in divisions:
        div = int(div)
        base = "object" if div == 1 else ("octant8" if div == 2 else f"grid{div**3}")
        result.append(Candidate(base, div, None))
        if div > 1:
            for kappa in shrinkage_kappas:
                result.append(Candidate(f"{base}_shrink_k{int(kappa)}", div, float(kappa)))
    names = [x.name for x in result]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate candidate names={names}")
    return tuple(result)
