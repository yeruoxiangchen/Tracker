from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

from pose_point_depth_mv.native_ss_occupancy import (
    coords_from_logits,
    frozen_decoder_occupancy_objective,
    logit_quantiles,
    target_occupancy_grid,
)


def test_target_occupancy_grid_uses_last_three_columns_and_filters_bounds() -> None:
    coords = torch.tensor(
        [
            [7, 1, 2, 3],
            [8, 1, 2, 3],
            [9, -1, 0, 0],
            [10, 4, 0, 0],
        ]
    )
    target = target_occupancy_grid(
        coords, device=torch.device("cpu"), resolution=4
    )
    assert target.shape == (1, 1, 4, 4, 4)
    assert target.dtype == torch.bool
    assert int(target.sum().item()) == 1
    assert bool(target[0, 0, 1, 2, 3].item())


def test_target_occupancy_grid_rejects_empty_valid_target() -> None:
    try:
        target_occupancy_grid(
            torch.tensor([[-1, 0, 0], [4, 0, 0]]),
            device=torch.device("cpu"),
            resolution=4,
        )
    except ValueError as error:
        assert "no valid occupied voxels" in str(error)
    else:
        raise AssertionError("empty target occupancy was accepted")


def test_coords_from_logits_uses_strict_common_threshold() -> None:
    logits = torch.full((1, 1, 2, 2, 2), -1.0)
    logits[0, 0, 0, 1, 1] = 0.0
    logits[0, 0, 1, 0, 1] = 0.25
    coords = coords_from_logits(logits, threshold=0.0)
    np.testing.assert_array_equal(coords, np.asarray([[0, 1, 0, 1]], np.int32))


def test_occupancy_objective_separates_fn_fp_and_stock_rank() -> None:
    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.bool)
    target[0, 0, 0, 0, 0] = True
    full = torch.full(target.shape, -1.0)
    full[0, 0, 0, 0, 0] = -1.0
    full[0, 0, 1, 1, 1] = 2.0
    stock = torch.full(target.shape, -1.0)
    stock[0, 0, 0, 0, 0] = 0.5

    values = frozen_decoder_occupancy_objective(
        full,
        target,
        stock_logits=stock,
    )

    assert math.isclose(float(values["false_negative_loss"].item()), 1.0)
    assert math.isclose(
        float(values["false_positive_loss"].item()), 2.0 / 7.0, rel_tol=1.0e-6
    )
    assert math.isclose(float(values["stock_recall_rank_loss"].item()), 1.5)
    assert float(values["full_target_recall"].item()) == 0.0
    assert math.isclose(
        float(values["full_false_positive_rate"].item()),
        1.0 / 7.0,
        rel_tol=1.0e-6,
    )
    assert float(values["stock_target_recall"].item()) == 1.0


def test_occupancy_objective_zero_for_threshold_correct_logits() -> None:
    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.bool)
    target[0, 0, 0, 0, 0] = True
    full = torch.full(target.shape, -1.0)
    full[target] = 1.0
    values = frozen_decoder_occupancy_objective(
        full,
        target,
        stock_logits=full.detach().clone(),
    )
    assert float(values["false_negative_loss"].item()) == 0.0
    assert float(values["false_positive_loss"].item()) == 0.0
    assert float(values["stock_recall_rank_loss"].item()) == 0.0
    assert float(values["full_target_recall"].item()) == 1.0
    assert float(values["full_false_positive_rate"].item()) == 0.0


def test_frozen_decoder_propagates_finite_gradient_only_to_latent() -> None:
    decoder = nn.Conv3d(1, 1, kernel_size=1, bias=False)
    decoder.weight.data.fill_(1.0)
    decoder.requires_grad_(False)
    latent = torch.full((1, 1, 2, 2, 2), -0.5, requires_grad=True)
    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.bool)
    target[0, 0, 0, 0, 0] = True
    full_logits = decoder(latent)
    values = frozen_decoder_occupancy_objective(full_logits, target)
    loss = values["false_negative_loss"] + values["false_positive_loss"]
    loss.backward()

    assert latent.grad is not None
    assert bool(torch.isfinite(latent.grad).all().item())
    assert float(latent.grad.abs().sum().item()) > 0.0
    assert decoder.weight.grad is None


def test_logit_quantiles_report_target_groups() -> None:
    logits = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
    target = torch.zeros_like(logits, dtype=torch.bool)
    target.reshape(-1)[:2] = True
    values = logit_quantiles(logits, target, (0.0, 0.5, 1.0))
    assert set(values) == {"all", "target_positive", "target_negative"}
    assert values["target_positive"]["0"] == 0.0
    assert values["target_positive"]["1"] == 1.0
    assert values["target_negative"]["0"] == 2.0
