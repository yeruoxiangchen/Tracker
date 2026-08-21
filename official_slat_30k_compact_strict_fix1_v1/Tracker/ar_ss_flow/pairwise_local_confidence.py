#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F

LOCAL_CONFIDENCE_METHODS = ("raw", "local_mean", "local_topk")


def infer_volume_side(voxel_count: int) -> int:
    side = int(round(float(voxel_count) ** (1.0 / 3.0)))
    if side <= 0 or side**3 != int(voxel_count):
        raise ValueError(f"voxel_count={voxel_count} is not a perfect cube")
    return side


def parse_methods(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        methods = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        methods = tuple(str(item).strip() for item in value if str(item).strip())
    if not methods:
        raise ValueError("at least one local confidence method is required")
    invalid = [method for method in methods if method not in LOCAL_CONFIDENCE_METHODS]
    if invalid:
        raise ValueError(f"invalid local confidence methods={invalid}")
    if len(set(methods)) != len(methods):
        raise ValueError(f"duplicate local confidence methods={methods}")
    return methods


def _as_volume_batch(values: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    if values.ndim == 1:
        values = values[None, :]
    if support.ndim == 1:
        support = support[None, :]
    if values.ndim != 2 or support.ndim != 2:
        raise ValueError("values and support must be [N] or [B,N]")
    if values.shape != support.shape:
        raise ValueError(f"values/support shape mismatch: {values.shape} != {support.shape}")
    side = infer_volume_side(int(values.shape[1]))
    values_5d = values.float().reshape(-1, 1, side, side, side)
    support_5d = support.float().reshape(-1, 1, side, side, side).clamp_min(0.0)
    return values_5d, support_5d, side


def support_aware_local_mean(
    values: torch.Tensor,
    support: torch.Tensor,
    *,
    radius: int = 1,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Common-support weighted local mean for flattened cubic volumes.

    The same support map should be used for correct/wrong/shuffle branches so
    local post-processing cannot introduce a branch-specific geometry shortcut.
    """
    if int(radius) < 0:
        raise ValueError("radius must be non-negative")
    if int(radius) == 0:
        return values.float().clone()
    values_5d, support_5d, _ = _as_volume_batch(values, support)
    kernel = 2 * int(radius) + 1
    numerator = F.avg_pool3d(
        values_5d * support_5d,
        kernel_size=kernel,
        stride=1,
        padding=int(radius),
        count_include_pad=True,
    ) * float(kernel**3)
    denominator = F.avg_pool3d(
        support_5d,
        kernel_size=kernel,
        stride=1,
        padding=int(radius),
        count_include_pad=True,
    ) * float(kernel**3)
    result = numerator / denominator.clamp_min(float(eps))
    result = result * denominator.gt(float(eps)).float()
    return result.reshape(values_5d.shape[0], -1)


def support_aware_local_topk(
    values: torch.Tensor,
    support: torch.Tensor,
    *,
    radius: int = 1,
    topk: int = 8,
) -> torch.Tensor:
    """Average the highest supported confidences inside a cubic neighborhood."""
    if int(radius) < 0:
        raise ValueError("radius must be non-negative")
    if int(topk) <= 0:
        raise ValueError("topk must be positive")
    if int(radius) == 0:
        return values.float().clone()
    values_5d, support_5d, side = _as_volume_batch(values, support)
    batch = int(values_5d.shape[0])
    pad = int(radius)
    padded_values = F.pad(values_5d, (pad, pad, pad, pad, pad, pad), value=0.0)
    padded_support = F.pad(support_5d, (pad, pad, pad, pad, pad, pad), value=0.0)
    candidates: list[torch.Tensor] = []
    candidate_valid: list[torch.Tensor] = []
    for dz in range(2 * pad + 1):
        for dy in range(2 * pad + 1):
            for dx in range(2 * pad + 1):
                candidates.append(
                    padded_values[
                        :,
                        :,
                        dz : dz + side,
                        dy : dy + side,
                        dx : dx + side,
                    ].reshape(batch, -1)
                )
                candidate_valid.append(
                    padded_support[
                        :,
                        :,
                        dz : dz + side,
                        dy : dy + side,
                        dx : dx + side,
                    ].reshape(batch, -1).gt(0.0)
                )
    stacked = torch.stack(candidates, dim=1)
    valid = torch.stack(candidate_valid, dim=1)
    masked = torch.where(valid, stacked, torch.full_like(stacked, -torch.inf))
    selected_count = min(int(topk), int(masked.shape[1]))
    selected = torch.topk(masked, k=selected_count, dim=1, largest=True).values
    selected_valid = torch.isfinite(selected)
    selected_sum = torch.where(selected_valid, selected, torch.zeros_like(selected)).sum(dim=1)
    selected_denominator = selected_valid.sum(dim=1)
    result = selected_sum / selected_denominator.clamp_min(1).float()
    result = result * selected_denominator.gt(0).float()
    return result


def transform_confidence(
    values: torch.Tensor,
    support: torch.Tensor,
    *,
    method: str,
    radius: int = 1,
    topk: int = 8,
) -> torch.Tensor:
    if method == "raw":
        if values.ndim == 1:
            return values.float()[None, :]
        if values.ndim != 2:
            raise ValueError("raw values must be [N] or [B,N]")
        return values.float().clone()
    if method == "local_mean":
        return support_aware_local_mean(values, support, radius=radius)
    if method == "local_topk":
        return support_aware_local_topk(values, support, radius=radius, topk=topk)
    raise ValueError(f"unknown local confidence method={method}")


def transform_confidence_batched(
    values: torch.Tensor,
    support: torch.Tensor,
    *,
    method: str,
    radius: int = 1,
    topk: int = 8,
    batch_size: int = 32,
) -> torch.Tensor:
    if values.ndim != 2 or support.ndim != 2:
        raise ValueError("batched transform expects [B,N]")
    if values.shape != support.shape:
        raise ValueError("values/support shape mismatch")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    outputs: list[torch.Tensor] = []
    for start in range(0, int(values.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(values.shape[0]))
        outputs.append(
            transform_confidence(
                values[start:end],
                support[start:end],
                method=method,
                radius=radius,
                topk=topk,
            )
        )
    return torch.cat(outputs, dim=0) if outputs else values.new_zeros(values.shape)
