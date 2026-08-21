from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from ar_ss_flow.local_pose_lifting_flow import (
    PoseLiftingCacheDataset,
    parse_indices,
)
from pose_point_depth_mv.geometry import EVIDENCE_NAMES


PPD_PROBE_CACHE_VERSION = "pose_point_depth_mv.probe_cache.v1"
PPD_LOCAL_TARGET_PROBE_VERSION = "pose_point_depth_mv.local_target_probe.v1"

PROBE_CORRUPTIONS: dict[str, dict[str, str]] = {
    "pose_cyclic1": {"pose_mode": "pose_cyclic1"},
    "pose_reverse": {"pose_mode": "pose_reverse"},
    "depth_view_cyclic1": {"depth_mode": "depth_view_cyclic1"},
    "depth_spatial": {"depth_mode": "depth_spatial"},
    "point_reflect": {"point_mode": "point_reflect"},
    "point_cross_object": {"point_mode": "point_cross_object"},
}

POSITIVE_INDEX = EVIDENCE_NAMES.index("positive_label")
NEGATIVE_INDEX = EVIDENCE_NAMES.index("negative_label")
NEUTRAL_INDEX = EVIDENCE_NAMES.index("neutral_label")
XYZ_INDICES = tuple(EVIDENCE_NAMES.index(name) for name in ("x", "y", "z"))
POINT_INDICES = tuple(
    EVIDENCE_NAMES.index(name)
    for name in ("prior_occupancy", "prior_confidence", "prior_distance")
)
POSE_DEPTH_INDICES = tuple(
    EVIDENCE_NAMES.index(name)
    for name in (
        "surface_support",
        "free_space_support",
        "occluded_support",
        "valid_view_fraction",
        "mask_view_fraction",
        "depth_confidence",
        "signed_depth_mean",
        "signed_depth_std",
    )
)
EVIDENCE_ABLATIONS = (
    "full",
    "active_mask_only",
    "point_only",
    "pose_depth_only",
)


def make_null_evidence(evidence: torch.Tensor) -> torch.Tensor:
    """Remove object evidence while preserving the canonical XYZ frame."""

    if evidence.ndim != 5 or evidence.shape[1] != len(EVIDENCE_NAMES):
        raise ValueError(
            f"evidence must be [B,{len(EVIDENCE_NAMES)},16,16,16]"
        )
    null = torch.zeros_like(evidence)
    null[:, NEUTRAL_INDEX] = 1.0
    for index in XYZ_INDICES:
        null[:, index] = evidence[:, index]
    return null


def ablate_evidence(
    evidence: torch.Tensor,
    mode: str,
    *,
    reference_active_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply a content ablation while keeping one shared support-mask base."""

    if evidence.ndim != 5 or evidence.shape[1] != len(EVIDENCE_NAMES):
        raise ValueError(
            f"evidence must be [B,{len(EVIDENCE_NAMES)},16,16,16]"
        )
    expected_mask = (evidence.shape[0], 1, *evidence.shape[-3:])
    if reference_active_mask.shape != expected_mask:
        raise ValueError(
            "reference_active_mask must have shape "
            f"{expected_mask}, got {tuple(reference_active_mask.shape)}"
        )
    if mode not in EVIDENCE_ABLATIONS:
        raise ValueError(
            f"unsupported evidence ablation={mode!r}; "
            f"expected one of {EVIDENCE_ABLATIONS}"
        )
    if mode == "full":
        return evidence

    active = (reference_active_mask > 0.5).to(dtype=evidence.dtype)
    output = torch.zeros_like(evidence)
    output[:, POSITIVE_INDEX : POSITIVE_INDEX + 1] = active
    output[:, NEUTRAL_INDEX : NEUTRAL_INDEX + 1] = 1.0 - active
    for index in XYZ_INDICES:
        output[:, index] = evidence[:, index]
    if mode == "point_only":
        for index in POINT_INDICES:
            output[:, index] = evidence[:, index]
    elif mode == "pose_depth_only":
        for index in POSE_DEPTH_INDICES:
            output[:, index] = evidence[:, index]
    return output


class PPDProbeEvidenceDataset(Dataset):
    """Join immutable SS targets/conditions with precomputed PPD evidence."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        indices: str = "all",
        eligible_only: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != PPD_PROBE_CACHE_VERSION:
            raise ValueError(
                f"unsupported PPD probe cache format={payload.get('format')!r}"
            )
        if tuple(payload.get("evidence_names", ())) != EVIDENCE_NAMES:
            raise ValueError("PPD evidence feature names do not match code")
        rows = payload.get("samples")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"PPD probe cache has no samples: {manifest}")
        selected = parse_indices(indices, len(rows))
        selected_rows = [rows[index] for index in selected]
        if eligible_only:
            selected_rows = [row for row in selected_rows if row.get("eligible")]
        if not selected_rows:
            raise ValueError("PPD probe dataset selection has no eligible samples")
        self.rows = selected_rows
        self.root = Path(payload.get("output_dir", self.manifest_path.parent))
        self.config_hash = str(payload.get("config_hash", ""))
        if not self.config_hash:
            raise ValueError("PPD probe cache is missing config_hash")
        self.source_manifest = Path(payload["source_cache_manifest"])
        self.source_dataset = PoseLiftingCacheDataset(
            self.source_manifest, indices="all"
        )
        self.corruption_names = tuple(payload["corruption_names"])
        if self.corruption_names != tuple(PROBE_CORRUPTIONS):
            raise ValueError(
                "PPD probe corruption schema mismatch: "
                f"{self.corruption_names} != {tuple(PROBE_CORRUPTIONS)}"
            )
        uids = [str(row["uid"]) for row in self.rows]
        if len(set(uids)) != len(uids):
            raise ValueError("PPD probe selection contains duplicate UIDs")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source_index = int(row["source_index"])
        source = self.source_dataset[source_index]
        if str(source["uid"]) != str(row["uid"]):
            raise ValueError(
                f"source UID mismatch: {source['uid']} != {row['uid']}"
            )
        evidence_path = Path(row["evidence_file"])
        if not evidence_path.is_absolute():
            evidence_path = self.root / evidence_path
        evidence = torch.load(evidence_path, map_location="cpu")
        if evidence.get("format") != PPD_PROBE_CACHE_VERSION:
            raise ValueError(f"invalid evidence file format: {evidence_path}")
        if str(evidence.get("config_hash", "")) != self.config_hash:
            raise ValueError(f"evidence config hash mismatch: {evidence_path}")
        if str(evidence.get("uid")) != str(row["uid"]):
            raise ValueError(f"evidence UID mismatch: {evidence_path}")
        correct = evidence["correct_features"]
        corruptions = evidence["corrupt_features"]
        expected = (len(EVIDENCE_NAMES), 16, 16, 16)
        if tuple(correct.shape) != expected:
            raise ValueError(
                f"uid={row['uid']} invalid correct evidence {tuple(correct.shape)}"
            )
        if not bool(torch.isfinite(correct).all().item()):
            raise ValueError(f"uid={row['uid']} correct evidence is non-finite")
        if tuple(corruptions) != self.corruption_names:
            raise ValueError(f"uid={row['uid']} corruption keys mismatch")
        for name, value in corruptions.items():
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"uid={row['uid']} corruption={name} shape={tuple(value.shape)}"
                )
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(
                    f"uid={row['uid']} corruption={name} is non-finite"
                )
        result = dict(source)
        result.update(
            {
                "ppd_source_index": source_index,
                "ppd_cache_path": str(evidence_path),
                "ppd_correct_features": correct.float(),
                "ppd_corrupt_features": {
                    name: value.float() for name, value in corruptions.items()
                },
                "ppd_stats": evidence["stats"],
                "ppd_calibration": evidence["calibration"],
            }
        )
        return result


class PPDLocalTargetProbe(nn.Module):
    """Low-rank same-voxel probe for ``v_gt - v_stock`` learnability."""

    def __init__(
        self,
        *,
        evidence_dim: int = len(EVIDENCE_NAMES),
        latent_channels: int = 8,
        rank: int = 32,
    ) -> None:
        super().__init__()
        self.evidence_dim = int(evidence_dim)
        self.latent_channels = int(latent_channels)
        self.rank = int(rank)
        if self.evidence_dim != len(EVIDENCE_NAMES):
            raise ValueError("PPD probe evidence dimension must match EVIDENCE_NAMES")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        state_channels = 2 * self.latent_channels + 1
        self.state_projection = nn.Conv3d(
            state_channels, self.rank, kernel_size=1, bias=False
        )
        self.evidence_projection = nn.Conv3d(
            self.evidence_dim, self.rank, kernel_size=1, bias=False
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(2 * self.rank, self.rank, kernel_size=1, bias=False),
            nn.SiLU(),
        )
        self.output = nn.Conv3d(
            self.rank, self.latent_channels, kernel_size=1, bias=False
        )
        nn.init.zeros_(self.output.weight)

    def metadata(self) -> dict[str, Any]:
        return {
            "version": PPD_LOCAL_TARGET_PROBE_VERSION,
            "type": type(self).__name__,
            "fusion": "low_rank_same_voxel_bilinear",
            "global_attention": False,
            "local_neighborhood": 1,
            "evidence_dim": self.evidence_dim,
            "evidence_names": list(EVIDENCE_NAMES),
            "latent_channels": self.latent_channels,
            "rank": self.rank,
            "time_normalization": "t_div_1000",
            "zero_init_output": True,
            "neutral_exact_zero": True,
            "physical_off_exact_zero": True,
            "null_evidence_preserves_xyz": True,
            "supports_active_mask_override": True,
            "evidence_ablation_modes": list(EVIDENCE_ABLATIONS),
            "uses_old_c2_residual": False,
        }

    @staticmethod
    def active_mask(evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 5 or evidence.shape[1] != len(EVIDENCE_NAMES):
            raise ValueError(
                f"evidence must be [B,{len(EVIDENCE_NAMES)},16,16,16]"
            )
        positive = evidence[:, POSITIVE_INDEX : POSITIVE_INDEX + 1] > 0.5
        negative = evidence[:, NEGATIVE_INDEX : NEGATIVE_INDEX + 1] > 0.5
        return (positive | negative).to(dtype=torch.float32)

    def forward(
        self,
        x_t: torch.Tensor,
        stock_velocity: torch.Tensor,
        t: torch.Tensor,
        evidence: torch.Tensor,
        *,
        physical_present: bool = True,
        scale: float = 1.0,
        active_mask_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected = (16, 16, 16)
        if x_t.shape != stock_velocity.shape:
            raise ValueError("x_t and stock velocity must have identical shapes")
        if x_t.ndim != 5 or x_t.shape[1] != self.latent_channels:
            raise ValueError("x_t must be [B,8,16,16,16]")
        if tuple(x_t.shape[-3:]) != expected:
            raise ValueError("PPD probe requires a 16^3 SS latent")
        if evidence.shape != (
            x_t.shape[0],
            self.evidence_dim,
            *expected,
        ):
            raise ValueError(f"invalid evidence shape={tuple(evidence.shape)}")
        if active_mask_override is None:
            active = self.active_mask(evidence).to(device=x_t.device)
        else:
            expected_mask = (x_t.shape[0], 1, *expected)
            if active_mask_override.shape != expected_mask:
                raise ValueError(
                    "active_mask_override must have shape "
                    f"{expected_mask}, got {tuple(active_mask_override.shape)}"
                )
            if not bool(torch.isfinite(active_mask_override).all().item()):
                raise ValueError("active_mask_override contains non-finite values")
            active = (active_mask_override.to(device=x_t.device) > 0.5).to(
                dtype=torch.float32
            )
        if not physical_present or float(scale) == 0.0:
            zero = torch.zeros_like(stock_velocity)
            return zero, self._stats(zero, active)
        batch = int(x_t.shape[0])
        t_channel = (t.float() / 1000.0).reshape(batch, 1, 1, 1, 1).expand(
            batch, 1, *expected
        )
        state = torch.cat(
            (x_t.float(), stock_velocity.float(), t_channel), dim=1
        )
        state_hidden = self.state_projection(state)
        evidence_hidden = self.evidence_projection(evidence.float())
        interaction = torch.cat(
            (evidence_hidden, state_hidden * evidence_hidden), dim=1
        )
        delta = self.output(self.fusion(interaction))
        delta = delta * active * float(scale)
        return delta.to(dtype=stock_velocity.dtype), self._stats(delta, active)

    @staticmethod
    def _stats(
        delta: torch.Tensor, active: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        neutral = 1.0 - active
        return {
            "delta_rms": delta.float().square().mean().sqrt(),
            "delta_abs_max": delta.float().abs().max(),
            "active_ratio": active.float().mean(),
            "neutral_abs_max": (delta.float() * neutral).abs().max(),
        }


def load_probe_state(
    probe: PPDLocalTargetProbe, state: dict[str, torch.Tensor]
) -> None:
    missing, unexpected = probe.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"probe state mismatch: missing={missing}, unexpected={unexpected}"
        )
