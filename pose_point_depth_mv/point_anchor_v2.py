from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from ar_ss_flow.local_pose_lifting_flow import (
    PoseLiftingCacheDataset,
    parse_indices,
)


POINT_ANCHOR_CACHE_VERSION = "pose_point_depth_mv.point_anchor_cache.v2"
POINT_ANCHOR_MODEL_VERSION = "pose_point_depth_mv.point_anchor_probe.v2"
POINT_ANCHOR_CHECKPOINT_VERSION = "pose_point_depth_mv.point_anchor_checkpoint.v2"

POINT_EVIDENCE_NAMES = (
    "prior_occupancy",
    "prior_confidence",
    "prior_distance",
    "active_label",
    "neutral_label",
    "x",
    "y",
    "z",
)
POINT_CONTROL_NAMES = (
    "point_reflect",
    "point_axis_cycle",
    "point_spatial_roll",
    "point_cross_object_matched",
    "point_drop",
    "constant_prior",
)

OCCUPANCY_INDEX = POINT_EVIDENCE_NAMES.index("prior_occupancy")
CONFIDENCE_INDEX = POINT_EVIDENCE_NAMES.index("prior_confidence")
DISTANCE_INDEX = POINT_EVIDENCE_NAMES.index("prior_distance")
ACTIVE_INDEX = POINT_EVIDENCE_NAMES.index("active_label")
NEUTRAL_INDEX = POINT_EVIDENCE_NAMES.index("neutral_label")
XYZ_INDICES = tuple(POINT_EVIDENCE_NAMES.index(name) for name in ("x", "y", "z"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def validate_points(
    coords: torch.Tensor,
    confidence: torch.Tensor,
    *,
    uid: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"uid={uid} point coords must be [N,3/4]")
    if confidence.ndim != 1 or len(confidence) != len(coords):
        raise ValueError(f"uid={uid} point confidence shape mismatch")
    raw_xyz = coords[:, -3:]
    if not bool(torch.isfinite(raw_xyz.float()).all().item()):
        raise ValueError(f"uid={uid} has non-finite point coordinates")
    if not bool(torch.equal(raw_xyz.float(), raw_xyz.float().round())):
        raise ValueError(f"uid={uid} has non-integral point coordinates")
    xyz = raw_xyz.to(dtype=torch.long)
    conf = confidence.to(dtype=torch.float32)
    in_bounds = ((xyz >= 0) & (xyz < 64)).all(dim=1)
    if not bool(torch.isfinite(conf).all().item()):
        raise ValueError(f"uid={uid} has non-finite point confidence")
    if not bool(((conf >= 0.0) & (conf <= 1.0)).all().item()):
        raise ValueError(f"uid={uid} has point confidence outside [0,1]")
    if not bool(in_bounds.all().item()):
        raise ValueError(f"uid={uid} has point coordinates outside [0,63]")
    if len(xyz) == 0:
        raise ValueError(f"uid={uid} has no sparse points")
    return xyz, conf


def transform_points(xyz: torch.Tensor, mode: str) -> torch.Tensor:
    output = xyz.clone()
    if mode == "correct":
        return output
    if mode == "point_reflect":
        output[:, 0] = 63 - output[:, 0]
        output[:, 2] = 63 - output[:, 2]
        return output
    if mode == "point_axis_cycle":
        return output[:, (1, 2, 0)]
    if mode == "point_spatial_roll":
        shift = output.new_tensor((32, 16, 24))
        return (output + shift) % 64
    raise ValueError(f"unsupported point transform={mode!r}")


def deterministic_subsample_points(
    xyz: torch.Tensor,
    confidence: torch.Tensor,
    count: int,
    *,
    uid: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(xyz) != len(confidence):
        raise ValueError("point subsampling input lengths differ")
    if count <= 0 or count > len(xyz):
        raise ValueError("point subsampling count is outside the input range")
    if count == len(xyz):
        return xyz.clone(), confidence.clone()
    generator = torch.Generator(device="cpu").manual_seed(
        deterministic_seed(uid, seed, "point_subsample", count)
    )
    indices = torch.randperm(len(xyz), generator=generator)[:count]
    return xyz[indices].clone(), confidence[indices].clone()


def match_cross_object_points(
    correct_xyz: torch.Tensor,
    correct_confidence: torch.Tensor,
    candidate_xyz: torch.Tensor,
    candidate_confidence: torch.Tensor,
    *,
    uid: str,
    candidate_uid: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Match point count and confidence multiset while retaining candidate geometry."""

    target_count = int(len(correct_xyz))
    candidate_count = int(len(candidate_xyz))
    if target_count <= 0 or candidate_count <= 0:
        raise ValueError("cross-object matching requires non-empty point sets")
    generator = torch.Generator(device="cpu").manual_seed(
        deterministic_seed(uid, candidate_uid, seed, "cross_match")
    )
    if candidate_count >= target_count:
        indices = torch.randperm(candidate_count, generator=generator)[:target_count]
        replacement = False
    else:
        repeats = (target_count + candidate_count - 1) // candidate_count
        blocks = [torch.randperm(candidate_count, generator=generator) for _ in range(repeats)]
        indices = torch.cat(blocks)[:target_count]
        replacement = True
    matched_xyz = candidate_xyz[indices].clone()

    # Assign the exact correct confidence multiset according to candidate confidence rank.
    candidate_rank = torch.argsort(candidate_confidence[indices], stable=True)
    correct_sorted = torch.sort(correct_confidence).values
    matched_confidence = torch.empty_like(correct_sorted)
    matched_confidence[candidate_rank] = correct_sorted
    report = {
        "correct_count": target_count,
        "candidate_count": candidate_count,
        "matched_count": int(len(matched_xyz)),
        "used_coordinate_replacement": replacement,
        "confidence_multiset_max_abs_diff": float(
            (torch.sort(matched_confidence).values - correct_sorted).abs().max().item()
        ),
    }
    return matched_xyz, matched_confidence, report


def point_volume(
    xyz64: torch.Tensor,
    confidence: torch.Tensor,
    *,
    side: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = xyz64.device
    occupancy = torch.zeros(side**3, device=device, dtype=torch.float32)
    confidence_grid = torch.zeros_like(occupancy)
    if len(xyz64) == 0:
        distance = torch.ones_like(occupancy)
        shape = (side, side, side)
        return occupancy.reshape(shape), confidence_grid.reshape(shape), distance.reshape(shape)
    xyz = torch.div(xyz64.long() * side, 64, rounding_mode="floor")
    if not bool((((xyz >= 0) & (xyz < side)).all()).item()):
        raise ValueError("quantized point coordinate is outside point volume")
    flat = xyz[:, 0] * side * side + xyz[:, 1] * side + xyz[:, 2]
    occupancy[flat] = 1.0
    confidence_grid.scatter_reduce_(
        0,
        flat,
        confidence.float().clamp(0.0, 1.0),
        reduce="amax",
        include_self=True,
    )
    axis = torch.arange(side, device=device, dtype=torch.float32)
    centers = torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    distance = torch.cdist(centers, xyz.float()).amin(dim=1)
    distance = distance / max(math.sqrt(3.0) * float(side - 1), 1.0)
    shape = (side, side, side)
    return (
        occupancy.reshape(shape),
        confidence_grid.reshape(shape),
        distance.clamp(0.0, 1.0).reshape(shape),
    )


def build_point_evidence(
    xyz64: torch.Tensor,
    confidence: torch.Tensor,
    *,
    reference_active_mask: torch.Tensor | None = None,
    radius_voxels: float = 1.5,
    side: int = 16,
) -> torch.Tensor:
    occupancy, confidence_grid, distance = point_volume(
        xyz64, confidence, side=side
    )
    if reference_active_mask is None:
        radius_normalized = float(radius_voxels) / max(
            math.sqrt(3.0) * float(side - 1), 1.0
        )
        active = distance <= radius_normalized
    else:
        if tuple(reference_active_mask.shape) != (side, side, side):
            raise ValueError("reference active mask has invalid shape")
        active = reference_active_mask > 0.5
    axis = (torch.arange(side, device=xyz64.device, dtype=torch.float32) + 0.5)
    axis = axis / float(side) * 2.0 - 1.0
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=0)
    evidence = torch.stack(
        (
            occupancy,
            confidence_grid,
            distance,
            active.float(),
            (~active).float(),
            xyz[0],
            xyz[1],
            xyz[2],
        ),
        dim=0,
    )
    expected = (len(POINT_EVIDENCE_NAMES), side, side, side)
    if tuple(evidence.shape) != expected:
        raise RuntimeError(f"unexpected point evidence shape={tuple(evidence.shape)}")
    if not bool(torch.isfinite(evidence).all().item()):
        raise RuntimeError("point evidence contains non-finite values")
    return evidence.float()


def make_drop_evidence(correct: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(correct)
    output[DISTANCE_INDEX] = 1.0
    output[ACTIVE_INDEX] = correct[ACTIVE_INDEX]
    output[NEUTRAL_INDEX] = correct[NEUTRAL_INDEX]
    for index in XYZ_INDICES:
        output[index] = correct[index]
    return output


def make_constant_evidence(correct: torch.Tensor) -> torch.Tensor:
    output = make_drop_evidence(correct)
    active = correct[ACTIVE_INDEX] > 0.5
    if not bool(active.any().item()):
        raise ValueError("constant prior requires a non-empty correct active mask")
    for index in (OCCUPANCY_INDEX, CONFIDENCE_INDEX, DISTANCE_INDEX):
        value = correct[index][active].mean()
        output[index][active] = value
    return output


def make_null_point_evidence(evidence: torch.Tensor) -> torch.Tensor:
    if evidence.ndim != 5 or evidence.shape[1] != len(POINT_EVIDENCE_NAMES):
        raise ValueError("point evidence must be [B,8,16,16,16]")
    output = torch.zeros_like(evidence)
    output[:, DISTANCE_INDEX] = 1.0
    output[:, NEUTRAL_INDEX] = 1.0
    for index in XYZ_INDICES:
        output[:, index] = evidence[:, index]
    return output


class PointAnchorCacheDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        *,
        indices: str = "all",
    ) -> None:
        self.manifest_path = Path(manifest)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != POINT_ANCHOR_CACHE_VERSION:
            raise ValueError("unsupported point-anchor cache format")
        if tuple(payload.get("evidence_names", ())) != POINT_EVIDENCE_NAMES:
            raise ValueError("point-anchor evidence schema mismatch")
        if tuple(payload.get("control_names", ())) != POINT_CONTROL_NAMES:
            raise ValueError("point-anchor control schema mismatch")
        rows = payload.get("samples")
        if not isinstance(rows, list) or not rows:
            raise ValueError("point-anchor cache has no samples")
        selected = parse_indices(indices, len(rows))
        self.rows = [rows[index] for index in selected]
        self.root = Path(payload.get("output_dir", self.manifest_path.parent))
        self.config_hash = str(payload.get("config_hash", ""))
        self.source_manifest = Path(payload["source_cache_manifest"])
        if sha256_file(self.source_manifest) != payload["source_manifest_sha256"]:
            raise RuntimeError("source manifest SHA256 changed after point cache build")
        self.source_dataset = PoseLiftingCacheDataset(self.source_manifest, indices="all")
        uids = [str(row["uid"]) for row in self.rows]
        if len(set(uids)) != len(uids):
            raise ValueError("point-anchor subset contains duplicate UIDs")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source_index = int(row["source_index"])
        source = self.source_dataset[source_index]
        if str(source["uid"]) != str(row["uid"]):
            raise RuntimeError("point-anchor source UID mismatch")
        if row_hash(self.source_dataset.rows[source_index]) != row["source_row_hash"]:
            raise RuntimeError("point-anchor source row changed after cache build")
        path = Path(row["evidence_file"])
        if not path.is_absolute():
            path = self.root / path
        payload = torch.load(path, map_location="cpu")
        if payload.get("format") != POINT_ANCHOR_CACHE_VERSION:
            raise ValueError(f"invalid point-anchor sample format: {path}")
        if str(payload.get("config_hash")) != self.config_hash:
            raise ValueError(f"point-anchor config hash mismatch: {path}")
        if str(payload.get("uid")) != str(row["uid"]):
            raise ValueError(f"point-anchor UID mismatch: {path}")
        correct = payload["correct_evidence"].float()
        controls = {name: value.float() for name, value in payload["controls"].items()}
        expected = (len(POINT_EVIDENCE_NAMES), 16, 16, 16)
        if tuple(correct.shape) != expected or tuple(controls) != POINT_CONTROL_NAMES:
            raise ValueError(f"point-anchor evidence schema invalid: {path}")
        for name, evidence in {"correct": correct, **controls}.items():
            if tuple(evidence.shape) != expected:
                raise ValueError(f"uid={row['uid']} {name} evidence shape invalid")
            if not bool(torch.isfinite(evidence).all().item()):
                raise ValueError(f"uid={row['uid']} {name} evidence non-finite")
            if not torch.equal(evidence[ACTIVE_INDEX], correct[ACTIVE_INDEX]):
                raise ValueError(f"uid={row['uid']} {name} active mask is not fixed")
        result = dict(source)
        result.update(
            {
                "point_source_index": source_index,
                "point_cache_path": str(path),
                "point_correct_evidence": correct,
                "point_control_evidence": controls,
                "point_stats": payload["stats"],
            }
        )
        return result


class PointAnchorProbe(nn.Module):
    def __init__(self, *, rank: int = 32, latent_channels: int = 8) -> None:
        super().__init__()
        self.rank = int(rank)
        self.latent_channels = int(latent_channels)
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        self.state_projection = nn.Conv3d(
            2 * self.latent_channels + 1, self.rank, kernel_size=1, bias=False
        )
        self.evidence_projection = nn.Conv3d(
            len(POINT_EVIDENCE_NAMES), self.rank, kernel_size=1, bias=False
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
            "version": POINT_ANCHOR_MODEL_VERSION,
            "type": type(self).__name__,
            "rank": self.rank,
            "latent_channels": self.latent_channels,
            "evidence_names": list(POINT_EVIDENCE_NAMES),
            "controls": list(POINT_CONTROL_NAMES),
            "fusion": "low_rank_same_voxel_bilinear",
            "time_normalization": "t_div_1000",
            "zero_init_output": True,
            "fixed_correct_point_mask": True,
            "non_anchor_exact_zero": True,
            "uses_pose_depth": False,
            "uses_flow_lora": False,
        }

    @staticmethod
    def active_mask(evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 5 or evidence.shape[1] != len(POINT_EVIDENCE_NAMES):
            raise ValueError("point evidence must be [B,8,16,16,16]")
        return (evidence[:, ACTIVE_INDEX : ACTIVE_INDEX + 1] > 0.5).float()

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
            raise ValueError("x_t and stock velocity shapes differ")
        if x_t.ndim != 5 or x_t.shape[1] != self.latent_channels:
            raise ValueError("x_t must be [B,8,16,16,16]")
        if tuple(x_t.shape[-3:]) != expected:
            raise ValueError("point-anchor probe requires a 16^3 SS latent")
        if evidence.shape != (x_t.shape[0], len(POINT_EVIDENCE_NAMES), *expected):
            raise ValueError("point-anchor evidence shape mismatch")
        active = (
            self.active_mask(evidence).to(device=x_t.device)
            if active_mask_override is None
            else (active_mask_override.to(device=x_t.device) > 0.5).float()
        )
        if active.shape != (x_t.shape[0], 1, *expected):
            raise ValueError("point-anchor active mask shape mismatch")
        if not physical_present or float(scale) == 0.0:
            zero = torch.zeros_like(stock_velocity)
            return zero, self._stats(zero, active)
        batch = int(x_t.shape[0])
        t_channel = (t.float() / 1000.0).reshape(batch, 1, 1, 1, 1).expand(
            batch, 1, *expected
        )
        state = torch.cat((x_t.float(), stock_velocity.float(), t_channel), dim=1)
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


def load_point_probe_state(
    probe: PointAnchorProbe, state: dict[str, torch.Tensor]
) -> None:
    missing, unexpected = probe.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"point probe state mismatch: missing={missing}, unexpected={unexpected}"
        )
