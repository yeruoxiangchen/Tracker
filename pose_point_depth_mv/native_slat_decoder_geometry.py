#!/usr/bin/env python3
"""Differentiable frozen-decoder geometry losses for Native-SLat training.

The released Mesh decoder stays frozen.  Gradients are retained only with
respect to the predicted SLat input.  The helper deliberately stops before
FlexiCubes extraction: sparse SDF/deformation/topology fields are continuous,
whereas final faces/components are not a stable per-step training objective.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


DECODER_GEOMETRY_LOSS_VERSION = (
    "pose_point_depth_mv.native_slat_decoder_geometry.v1"
)


def decode_mesh_fields(decoder: Any, latent: Any) -> Any:
    """Run ``SLatMeshDecoder`` up to, but not through, Mesh extraction."""

    # Calling the base implementation explicitly is the external equivalent
    # of ``super().forward`` in SLatMeshDecoder.forward.  Keeping this helper in
    # the project avoids changing the frozen ReconViaGen dependency.
    from trellis.models.structured_latent_vae.base import SparseTransformerBase

    h = SparseTransformerBase.forward(decoder, latent)
    for block in decoder.upsample:
        h = block(h)
    h = h.type(latent.dtype)
    return decoder.out_layer(h)


def denormalize_sparse(latent: Any, mean: torch.Tensor, std: torch.Tensor) -> Any:
    if latent.feats.ndim != 2 or latent.feats.shape[-1] != mean.shape[-1]:
        raise ValueError("SLat/normalization channel mismatch")
    return latent.replace(
        latent.feats * std.to(latent.feats) + mean.to(latent.feats)
    )


def x0_from_velocity(sampler: Any, x_t: Any, t_value: float, velocity: Any) -> Any:
    prediction, _ = sampler._v_to_xstart_eps(x_t, float(t_value), velocity)
    if not torch.equal(prediction.coords, x_t.coords):
        raise RuntimeError("velocity-to-x0 conversion changed sparse coordinates")
    return prediction


def split_mesh_fields(decoder: Any, fields: Any) -> dict[str, torch.Tensor]:
    extractor = decoder.mesh_extractor
    result = {
        name: extractor.get_layout(fields.feats, name)
        for name in ("sdf", "deform", "weights")
    }
    if any(value is None for value in result.values()):
        raise RuntimeError("frozen Mesh decoder lacks required geometry layouts")
    result["sdf"] = result["sdf"].float() + float(extractor.sdf_bias)
    result["deform"] = result["deform"].float()
    result["weights"] = result["weights"].float()
    return result


def _weighted_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(
        prediction.float(), target.float(), reduction="none", beta=float(beta)
    )
    while weight.ndim < loss.ndim:
        weight = weight.unsqueeze(-1)
    expanded = weight.expand_as(loss)
    return (loss * expanded).sum() / expanded.sum().clamp_min(1.0)


def decoder_field_distance(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    boundary_tau: float = 0.02,
    boundary_floor: float = 0.05,
    sdf_beta: float = 0.005,
    sign_margin: float = 0.01,
    deform_weight: float = 0.10,
    topology_weight: float = 0.02,
    sign_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Compare continuous decoder fields with extra weight near the surface."""

    if boundary_tau <= 0.0 or sdf_beta <= 0.0 or sign_margin <= 0.0:
        raise ValueError("decoder geometry scales must be positive")
    if not 0.0 <= boundary_floor <= 1.0:
        raise ValueError("boundary_floor must be in [0,1]")
    for name in ("sdf", "deform", "weights"):
        if prediction[name].shape != target[name].shape:
            raise ValueError(
                f"decoder field shape mismatch for {name}: "
                f"{tuple(prediction[name].shape)} != {tuple(target[name].shape)}"
            )

    target_sdf = target["sdf"].float()
    predicted_sdf = prediction["sdf"].float()
    boundary = torch.exp(-target_sdf.abs() / float(boundary_tau))
    boundary = float(boundary_floor) + (1.0 - float(boundary_floor)) * boundary
    sdf = _weighted_smooth_l1(
        predicted_sdf, target_sdf, boundary, beta=float(sdf_beta)
    )

    # Zero for an exact target, while penalizing new sign crossings and missing
    # target crossings without pushing every SDF value to an arbitrary margin.
    target_sign = torch.where(
        target_sdf >= 0.0,
        torch.ones_like(target_sdf),
        -torch.ones_like(target_sdf),
    )
    required_signed = target_sdf.abs().clamp_max(float(sign_margin))
    sign = F.relu(required_signed - target_sign * predicted_sdf).mean()

    deform = F.smooth_l1_loss(
        prediction["deform"].float(), target["deform"].float(), beta=0.01
    )
    topology = F.smooth_l1_loss(
        prediction["weights"].float(), target["weights"].float(), beta=0.01
    )
    total = (
        sdf
        + float(sign_weight) * sign
        + float(deform_weight) * deform
        + float(topology_weight) * topology
    )
    return {
        "total": total,
        "sdf": sdf,
        "sign": sign,
        "deform": deform,
        "topology": topology,
        "boundary_weight_mean": boundary.mean(),
    }


def stock_relative_trust_loss(
    full_distance: torch.Tensor,
    stock_distance: torch.Tensor,
    *,
    required_improvement: float = 0.01,
    eps: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hinge requiring Full decoder fields to improve over frozen Stock."""

    if required_improvement < 0.0:
        raise ValueError("required_improvement must be non-negative")
    if eps <= 0.0:
        raise ValueError("Stock trust reference floor must be positive")
    reference = stock_distance.detach().clamp_min(float(eps))
    relative_delta = (full_distance - reference) / reference
    loss = F.relu(relative_delta + float(required_improvement))
    return loss, relative_delta


def decoder_geometry_objective(
    *,
    decoder: Any,
    sampler: Any,
    x_t: Any,
    t_value: float,
    full_velocity: Any,
    stock_velocity: Any,
    target: Any,
    mean: torch.Tensor,
    std: torch.Tensor,
    required_improvement: float,
) -> dict[str, torch.Tensor]:
    """Compute Full field distance and a matched Stock-relative trust hinge."""

    full_x0 = denormalize_sparse(
        x0_from_velocity(sampler, x_t, t_value, full_velocity), mean, std
    )
    with torch.no_grad():
        stock_x0 = denormalize_sparse(
            x0_from_velocity(sampler, x_t, t_value, stock_velocity), mean, std
        )
        target_x0 = denormalize_sparse(target, mean, std)
        stock_sparse = decode_mesh_fields(decoder, stock_x0)
        target_sparse = decode_mesh_fields(decoder, target_x0)
        stock_fields = split_mesh_fields(decoder, stock_sparse)
        target_fields = split_mesh_fields(decoder, target_sparse)
    full_sparse = decode_mesh_fields(decoder, full_x0)
    if not torch.equal(full_sparse.coords, target_sparse.coords) or not torch.equal(
        stock_sparse.coords, target_sparse.coords
    ):
        raise RuntimeError("decoder fields diverged on matched GT support")
    full_fields = split_mesh_fields(decoder, full_sparse)
    full = decoder_field_distance(full_fields, target_fields)
    with torch.no_grad():
        stock = decoder_field_distance(stock_fields, target_fields)
    trust, relative_delta = stock_relative_trust_loss(
        full["total"],
        stock["total"],
        required_improvement=float(required_improvement),
    )
    return {
        "field_loss": full["total"],
        "stock_field_loss": stock["total"],
        "trust_loss": trust,
        "relative_field_delta": relative_delta,
        "sdf_loss": full["sdf"],
        "sign_loss": full["sign"],
        "deform_loss": full["deform"],
        "topology_loss": full["topology"],
        "boundary_weight_mean": full["boundary_weight_mean"],
    }


__all__ = [
    "DECODER_GEOMETRY_LOSS_VERSION",
    "decode_mesh_fields",
    "decoder_field_distance",
    "decoder_geometry_objective",
    "denormalize_sparse",
    "split_mesh_fields",
    "stock_relative_trust_loss",
    "x0_from_velocity",
]
