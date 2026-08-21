#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Statistic:
    name: str
    kind: str
    fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "fraction": self.fraction,
        }


DEFAULT_STATISTICS = (
    "mean",
    "median",
    "trimmed_mean_10",
    "top20_mean",
)


def parse_statistics(value: str | Iterable[str]) -> list[Statistic]:
    if isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    else:
        names = [str(item).strip() for item in value if str(item).strip()]
    if not names:
        raise ValueError("statistics must be non-empty")

    result: list[Statistic] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        if name == "mean":
            result.append(Statistic(name=name, kind="mean"))
        elif name == "median":
            result.append(Statistic(name=name, kind="median"))
        elif name.startswith("trimmed_mean_"):
            percent = float(name.rsplit("_", 1)[1])
            fraction = percent / 100.0
            if not 0.0 <= fraction < 0.5:
                raise ValueError(f"invalid trimmed fraction in {name}")
            result.append(Statistic(name=name, kind="trimmed_mean", fraction=fraction))
        elif name.startswith("top") and name.endswith("_mean"):
            percent = float(name[3:-5])
            fraction = percent / 100.0
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"invalid top fraction in {name}")
            result.append(Statistic(name=name, kind="top_mean", fraction=fraction))
        else:
            raise ValueError(f"unsupported statistic={name}")
    return result


def statistic_from_dict(data: dict[str, Any]) -> Statistic:
    return Statistic(
        name=str(data["name"]),
        kind=str(data["kind"]),
        fraction=None if data.get("fraction") is None else float(data["fraction"]),
    )


def finite_supported_values(confidence: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(confidence, dtype=np.float64).reshape(-1)
    support = np.asarray(valid, dtype=bool).reshape(-1)
    if values.shape != support.shape:
        raise ValueError(f"confidence/support shape mismatch: {values.shape} vs {support.shape}")
    mask = support & np.isfinite(values)
    return values[mask]


def object_score(
    confidence: np.ndarray,
    valid: np.ndarray,
    statistic: Statistic,
    *,
    min_valid_voxels: int = 1,
) -> float:
    values = finite_supported_values(confidence, valid)
    if values.size < int(min_valid_voxels):
        return float("nan")
    if statistic.kind == "mean":
        return float(np.mean(values))
    if statistic.kind == "median":
        return float(np.median(values))
    if statistic.kind == "trimmed_mean":
        fraction = float(statistic.fraction or 0.0)
        ordered = np.sort(values)
        trim = int(np.floor(ordered.size * fraction))
        if trim > 0 and 2 * trim < ordered.size:
            ordered = ordered[trim:-trim]
        return float(np.mean(ordered))
    if statistic.kind == "top_mean":
        fraction = float(statistic.fraction or 0.0)
        count = max(1, int(np.ceil(values.size * fraction)))
        if count >= values.size:
            return float(np.mean(values))
        split = values.size - count
        top = np.partition(values, split)[split:]
        return float(np.mean(top))
    raise ValueError(f"unsupported statistic kind={statistic.kind}")


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
        average_rank = 0.5 * ((cursor + 1) + end)
        ranks[order[cursor:end]] = average_rank
        cursor = end
    rank_sum_positive = float(ranks[labels == 1].sum())
    return float(
        (rank_sum_positive - positive.size * (positive.size + 1) / 2.0)
        / (positive.size * negative.size)
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
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
    return ranks


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 3:
        return 0.0
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    if np.std(left_rank) <= 1.0e-12 or np.std(right_rank) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def candidate_thresholds(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    values = np.concatenate(
        [
            np.asarray(positive, dtype=np.float64).reshape(-1),
            np.asarray(negative, dtype=np.float64).reshape(-1),
        ]
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([0.5], dtype=np.float64)
    candidates = np.quantile(values, np.linspace(0.01, 0.99, 99))
    return np.unique(candidates)


def choose_threshold(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    min_correct_coverage: float,
    max_correct_coverage: float,
) -> dict[str, float]:
    positive = np.asarray(positive, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative, dtype=np.float64).reshape(-1)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("positive and negative scores must be non-empty")

    best: tuple[float, float, float, float] | None = None
    fallback: tuple[float, float, float, float] | None = None
    for threshold in candidate_thresholds(positive, negative):
        tpr = float(np.mean(positive >= threshold))
        fpr = float(np.mean(negative >= threshold))
        youden = tpr - fpr
        balanced = 0.5 * (tpr + (1.0 - fpr))
        row = (youden, balanced, -abs(tpr - 0.75), float(threshold))
        if fallback is None or row > fallback:
            fallback = row
        if float(min_correct_coverage) <= tpr <= float(max_correct_coverage):
            if best is None or row > best:
                best = row
    chosen = best if best is not None else fallback
    if chosen is None:
        raise RuntimeError("threshold search produced no candidates")
    threshold = float(chosen[3])
    tpr = float(np.mean(positive >= threshold))
    fpr = float(np.mean(negative >= threshold))
    return {
        "threshold": threshold,
        "correct_coverage": tpr,
        "negative_coverage": fpr,
        "youden": tpr - fpr,
        "balanced_accuracy": 0.5 * (tpr + (1.0 - fpr)),
    }


def scalar_gate(score: np.ndarray | float, tau_low: float, tau_high: float):
    if not np.isfinite(tau_low) or not np.isfinite(tau_high):
        raise ValueError("gate thresholds must be finite")
    if tau_high <= tau_low:
        raise ValueError("tau_high must be greater than tau_low")
    value = np.asarray(score, dtype=np.float64)
    gate = np.clip((value - float(tau_low)) / (float(tau_high) - float(tau_low)), 0.0, 1.0)
    if np.isscalar(score):
        return float(gate)
    return gate


def calibrate_thresholds(
    correct: np.ndarray,
    negative: np.ndarray,
    *,
    min_correct_coverage: float,
    max_correct_coverage: float,
    tau_high_quantile: float,
    minimum_width: float = 1.0e-3,
) -> dict[str, float]:
    selected = choose_threshold(
        correct,
        negative,
        min_correct_coverage=min_correct_coverage,
        max_correct_coverage=max_correct_coverage,
    )
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    correct = correct[np.isfinite(correct)]
    tau_low = float(selected["threshold"])
    tau_high = float(np.quantile(correct, float(tau_high_quantile)))
    tau_high = max(tau_high, tau_low + float(minimum_width))
    return {
        **selected,
        "tau_low": tau_low,
        "tau_high": tau_high,
        "tau_high_quantile": float(tau_high_quantile),
    }


def load_calibration(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def compute_gate_from_calibration(
    confidence: np.ndarray,
    valid: np.ndarray,
    calibration: dict[str, Any],
) -> tuple[float, float]:
    statistic = statistic_from_dict(calibration["statistic"])
    score = object_score(
        confidence,
        valid,
        statistic,
        min_valid_voxels=int(calibration.get("minimum_valid_voxels", 1)),
    )
    gate = scalar_gate(
        score,
        float(calibration["tau_low"]),
        float(calibration["tau_high"]),
    )
    return score, gate
