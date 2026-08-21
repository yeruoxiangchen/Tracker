from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


NATIVE_SS_OCCUPANCY_OBJECTIVE_VERSION = (
    "pose_point_depth_mv.native_ss_occupancy_objective.v1"
)


def target_occupancy_grid(
    target_coords: torch.Tensor,
    *,
    device: torch.device,
    resolution: int = 64,
) -> torch.Tensor:
    side = int(resolution)
    if side <= 0:
        raise ValueError("occupancy resolution must be positive")
    coords = target_coords.detach().to(device=device, dtype=torch.long)[..., -3:]
    if coords.ndim != 2 or int(coords.shape[1]) != 3:
        raise ValueError("target occupancy coordinates must have shape [N,3] or [N,>=3]")
    valid = ((coords >= 0) & (coords < side)).all(dim=1)
    coords = coords[valid]
    target = torch.zeros(
        (1, 1, side, side, side), device=device, dtype=torch.bool
    )
    if coords.numel():
        target[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = True
    if not bool(target.any().item()):
        raise ValueError("target occupancy contains no valid occupied voxels")
    return target


def coords_from_logits(
    logits: torch.Tensor,
    *,
    threshold: float = 0.0,
) -> np.ndarray:
    value = float(threshold)
    if not math.isfinite(value):
        raise ValueError("occupancy threshold must be finite")
    if logits.ndim != 5 or int(logits.shape[0]) != 1 or int(logits.shape[1]) != 1:
        raise ValueError("occupancy logits must have shape [1,1,D,H,W]")
    return (
        torch.argwhere(logits.float() > value)[:, [0, 2, 3, 4]]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32)
    )


def frozen_decoder_occupancy_objective(
    full_logits: torch.Tensor,
    target_occupancy: torch.Tensor,
    *,
    stock_logits: torch.Tensor | None = None,
    false_negative_margin: float = 0.0,
    stock_recall_margin: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Return threshold-aligned losses while leaving decoder parameters frozen."""

    if full_logits.shape != target_occupancy.shape:
        raise ValueError(
            "full logits/target occupancy shapes differ: "
            f"{tuple(full_logits.shape)} != {tuple(target_occupancy.shape)}"
        )
    if stock_logits is not None and stock_logits.shape != full_logits.shape:
        raise ValueError("stock/full occupancy logit shapes differ")
    fn_margin = float(false_negative_margin)
    stock_margin = float(stock_recall_margin)
    if (
        not math.isfinite(fn_margin)
        or fn_margin < 0.0
        or not math.isfinite(stock_margin)
        or stock_margin < 0.0
    ):
        raise ValueError("occupancy margins must be finite and non-negative")

    target = target_occupancy.bool()
    positive_logits = full_logits.float()[target]
    negative_logits = full_logits.float()[~target]
    if positive_logits.numel() == 0 or negative_logits.numel() == 0:
        raise ValueError("occupancy objective requires positive and negative voxels")

    false_negative = F.relu(fn_margin - positive_logits).mean()
    false_positive = F.relu(negative_logits + fn_margin).mean()
    stock_recall_rank = false_negative.new_zeros(())
    stock_positive_logits = None
    if stock_logits is not None:
        stock_positive_logits = stock_logits.detach().float()[target]
        stock_recall_rank = F.relu(
            stock_positive_logits + stock_margin - positive_logits
        ).mean()

    with torch.no_grad():
        full_positive = positive_logits > 0.0
        full_negative = negative_logits > 0.0
        full_count = (full_logits.float() > 0.0).sum()
        target_count = target.sum()
        result: dict[str, torch.Tensor] = {
            "false_negative_loss": false_negative,
            "false_positive_loss": false_positive,
            "stock_recall_rank_loss": stock_recall_rank,
            "full_target_recall": full_positive.float().mean(),
            "full_false_positive_rate": full_negative.float().mean(),
            "full_positive_logit_mean": positive_logits.mean(),
            "full_negative_logit_mean": negative_logits.mean(),
            "full_occupied_count": full_count.to(torch.float32),
            "target_occupied_count": target_count.to(torch.float32),
            "full_target_count_ratio": full_count.float()
            / target_count.float().clamp_min(1.0),
        }
        if stock_positive_logits is not None and stock_logits is not None:
            stock_count = (stock_logits.detach().float() > 0.0).sum()
            result.update(
                {
                    "stock_target_recall": (
                        stock_positive_logits > 0.0
                    ).float().mean(),
                    "stock_positive_logit_mean": stock_positive_logits.mean(),
                    "stock_occupied_count": stock_count.to(torch.float32),
                    "full_minus_stock_occupied_count": (
                        full_count - stock_count
                    ).to(torch.float32),
                    "full_stock_count_ratio": full_count.float()
                    / stock_count.float().clamp_min(1.0),
                }
            )
    result.update(
        {
            "false_negative_loss": false_negative,
            "false_positive_loss": false_positive,
            "stock_recall_rank_loss": stock_recall_rank,
        }
    )
    return result


def logit_quantiles(
    logits: torch.Tensor,
    target_occupancy: torch.Tensor,
    quantiles: Iterable[float],
) -> dict[str, dict[str, float]]:
    if logits.shape != target_occupancy.shape:
        raise ValueError("logit quantile target shape differs")
    values = tuple(float(value) for value in quantiles)
    if (
        not values
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError("logit quantiles must be unique values in [0,1]")
    target = target_occupancy.bool()
    groups = {
        "all": logits.detach().float().reshape(-1),
        "target_positive": logits.detach().float()[target],
        "target_negative": logits.detach().float()[~target],
    }
    result: dict[str, dict[str, float]] = {}
    q = torch.tensor(values, device=logits.device, dtype=torch.float32)
    for name, group in groups.items():
        if group.numel() == 0:
            raise ValueError(f"logit quantile group is empty: {name}")
        measured = torch.quantile(group, q).cpu().tolist()
        result[name] = {
            f"{value:.6g}": float(item) for value, item in zip(values, measured)
        }
    return result


def objective_scalars(values: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: float(value.detach().float().item())
        for name, value in values.items()
        if torch.is_tensor(value) and value.numel() == 1
    }
