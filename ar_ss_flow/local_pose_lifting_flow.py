from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from ar_ss_flow.pose_lifting import (
    LIFTING_CACHE_VERSION,
    LIFTING_METADATA_NAMES,
    SUPPORT_METADATA_INDEX,
    build_lifting_volume,
    schema_hash,
)
from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256


LOCAL_LIFTING_ADAPTER_VERSION = "ar_ss_flow.local_pose_lifting_velocity.v2"


def parse_indices(spec: str, size: int) -> list[int]:
    text = str(spec).strip().lower()
    if text in {"", "all"}:
        return list(range(size))
    output: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            output.extend(range(int(start), int(end) + 1))
        else:
            output.append(int(item))
    bad = [index for index in output if index < 0 or index >= size]
    if bad:
        raise IndexError(f"indices outside size={size}: {bad}")
    return output


class PoseLiftingCacheDataset(Dataset):
    def __init__(self, manifest: str | Path, *, indices: str = "all") -> None:
        self.manifest_path = Path(manifest)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != LIFTING_CACHE_VERSION:
            raise ValueError(
                f"unsupported lifting cache format={payload.get('format')!r}"
            )
        if tuple(payload.get("metadata_names", ())) != LIFTING_METADATA_NAMES:
            raise ValueError("lifting metadata names do not match code")
        if payload.get("metadata_schema_hash") != schema_hash():
            raise ValueError("lifting metadata schema hash mismatch")
        rows = payload.get("samples")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"lifting cache has no samples: {manifest}")
        selected = parse_indices(indices, len(rows))
        self.rows = [rows[index] for index in selected]
        self.root = Path(payload.get("output_dir", self.manifest_path.parent))
        self.visual_feature_dim = int(payload["visual_feature_dim"])
        self.feature_metadata = dict(payload["feature_metadata"])
        self.config = dict(payload.get("config", {}))
        self.config_hash = str(payload.get("config_hash", ""))
        if not self.config_hash:
            raise ValueError("lifting cache manifest is missing config_hash")
        self.source_cache_manifest = str(payload.get("source_cache_manifest", ""))
        uids = [str(row.get("uid", "")) for row in self.rows]
        if not all(uids) or len(set(uids)) != len(uids):
            raise ValueError("lifting cache subset contains empty or duplicate uids")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = Path(row["cache_file"])
        if not path.is_absolute():
            path = self.root / path
        payload = torch.load(path, map_location="cpu")
        if payload.get("format") != LIFTING_CACHE_VERSION:
            raise ValueError(f"uid={row['uid']} invalid sample cache format")
        if str(payload.get("uid")) != str(row["uid"]):
            raise ValueError(f"cache uid mismatch: {payload.get('uid')} != {row['uid']}")
        visual = payload["visual_patch_features"]
        depth = payload["predicted_depth"]
        depth_confidence = payload["depth_confidence"]
        masks = payload["masks"]
        intrinsics = payload["intrinsics"]
        extrinsics = payload["extrinsics"]
        prior_coords = payload["prior_coords"]
        prior_confidence = payload["prior_confidence"]
        stock_condition = payload["stock_condition"]
        views = int(visual.shape[0])
        if visual.ndim != 3 or int(visual.shape[-1]) != self.visual_feature_dim:
            raise ValueError(f"uid={row['uid']} invalid visual shape {tuple(visual.shape)}")
        if depth.ndim != 3 or depth_confidence.shape != depth.shape or masks.shape != depth.shape:
            raise ValueError(f"uid={row['uid']} invalid depth/mask shapes")
        if intrinsics.shape != (views, 3, 3) or extrinsics.shape != (views, 4, 4):
            raise ValueError(f"uid={row['uid']} invalid K/T shapes")
        if prior_coords.ndim != 2 or prior_coords.shape[1] not in (3, 4):
            raise ValueError(f"uid={row['uid']} invalid prior coords")
        if prior_confidence.ndim != 1 or len(prior_confidence) != len(prior_coords):
            raise ValueError(f"uid={row['uid']} invalid prior confidence")
        if stock_condition.ndim != 3 or stock_condition.shape[0] != 1:
            raise ValueError(f"uid={row['uid']} invalid stock condition shape")
        shared = dict(payload.get("preprocessing", {}).get("shared_geometry", {}))
        if shared:
            if shared != dict(self.config.get("geometric_preprocessing", {})):
                raise ValueError(f"uid={row['uid']} sample/manifest preprocessing differs")
            source_intrinsics = payload.get("source_intrinsics")
            affines = payload.get("source_to_feature_affines")
            if not torch.is_tensor(source_intrinsics) or not torch.is_tensor(affines):
                raise ValueError(f"uid={row['uid']} shared geometry lacks K/A tensors")
            if source_intrinsics.shape != (views, 3, 3) or affines.shape != (views, 3, 3):
                raise ValueError(f"uid={row['uid']} invalid shared K/A shapes")
            expected_intrinsics = torch.matmul(
                affines.float(), source_intrinsics.float()
            )
            if not torch.allclose(
                intrinsics.float(), expected_intrinsics, rtol=1.0e-5, atol=1.0e-4
            ):
                raise ValueError(f"uid={row['uid']} intrinsics do not satisfy K'=A@K")
            identity = {
                "shared_geometry_hash": payload["preprocessing"].get(
                    "shared_geometry_hash"
                ),
                "view_ids": payload["view_ids"].to(torch.int64).tolist(),
                "source_intrinsics": source_intrinsics.float().tolist(),
                "feature_intrinsics": intrinsics.float().tolist(),
            }
            if canonical_json_sha256(identity) != payload["preprocessing"].get(
                "sample_geometry_identity_hash"
            ):
                raise ValueError(f"uid={row['uid']} sample geometry identity hash mismatch")
        tensors = (
            visual,
            depth,
            depth_confidence,
            masks,
            intrinsics,
            extrinsics,
            prior_confidence,
            stock_condition,
        )
        if not all(bool(torch.isfinite(value.float()).all().item()) for value in tensors):
            raise ValueError(f"uid={row['uid']} cache contains non-finite tensors")
        latent_path = Path(payload["ss_latent"])
        with np.load(latent_path) as latent:
            target = np.asarray(latent["z"], dtype=np.float32)
            target_coords = np.asarray(latent["target_coords"], dtype=np.int64)[:, -3:]
        if target.ndim == 5 and target.shape[0] == 1:
            target = target[0]
        if target.shape != (8, 16, 16, 16):
            raise ValueError(f"uid={row['uid']} invalid target shape {target.shape}")
        result = dict(payload)
        result.update(
            {
                "cache_path": str(path),
                "target": torch.from_numpy(target),
                "target_coords": torch.from_numpy(target_coords),
            }
        )
        return result


def collate_one(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 1:
        raise ValueError("pose lifting currently requires per-rank batch size 1")
    return rows[0]


def volume_from_sample(
    sample: dict[str, Any],
    *,
    device: torch.device,
    mode: str,
    compute_cross_view_metrics: bool = False,
    extrinsics_override: torch.Tensor | None = None,
    calibration_override: dict[str, Any] | None = None,
    object_to_world: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    cached = (
        sample.get("correct_geometry")
        if mode == "correct"
        and extrinsics_override is None
        and object_to_world is None
        else None
    )
    volume, metadata, stats = build_lifting_volume(
        visual_patch_features=sample["visual_patch_features"].to(
            device=device, dtype=torch.float16
        ),
        predicted_depth=sample["predicted_depth"].to(device=device),
        depth_confidence=sample["depth_confidence"].to(device=device),
        masks=sample["masks"].to(device=device),
        intrinsics=sample["intrinsics"].to(device=device),
        extrinsics=(
            extrinsics_override.to(device=device)
            if extrinsics_override is not None
            else sample["extrinsics"].to(device=device)
        ),
        prior_coords=sample["prior_coords"].to(device=device),
        prior_confidence=sample["prior_confidence"].to(device=device),
        calibration=(
            calibration_override
            if calibration_override is not None
            else sample["depth_calibration"]
        ),
        grid_transform=str(sample["grid_transform"]),
        extrinsics_type=str(sample["extrinsics_type"]),
        camera_forward_sign=float(sample["camera_forward_sign"]),
        pose_mode=mode,
        cached_correct_geometry=cached,
        compute_cross_view_metrics=compute_cross_view_metrics,
        object_to_world=object_to_world,
    )
    return volume.unsqueeze(0), metadata.unsqueeze(0), stats


class LocalPoseLiftingVelocityAdapter(nn.Module):
    """Same-voxel 16^3 residual; no all-to-all spatial attention is used."""

    def __init__(
        self,
        *,
        visual_channels: int,
        latent_channels: int = 8,
        hidden_dim: int = 96,
        metadata_channels: int = len(LIFTING_METADATA_NAMES),
    ) -> None:
        super().__init__()
        self.visual_channels = int(visual_channels)
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.metadata_channels = int(metadata_channels)
        self.state_projection = nn.Conv3d(2 * latent_channels + 1, hidden_dim, 1)
        self.visual_projection = nn.Conv3d(visual_channels, hidden_dim, 1, bias=False)
        self.metadata_projection = nn.Conv3d(metadata_channels, hidden_dim, 1)
        self.fusion = nn.Sequential(
            nn.Conv3d(5 * hidden_dim, 2 * hidden_dim, 1),
            nn.SiLU(),
            nn.Conv3d(2 * hidden_dim, hidden_dim, 1),
            nn.SiLU(),
        )
        self.output = nn.Conv3d(hidden_dim, latent_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def metadata(self) -> dict[str, Any]:
        return {
            "version": LOCAL_LIFTING_ADAPTER_VERSION,
            "fusion": "same_voxel_16x16x16_pointwise_mlp",
            "global_attention": False,
            "local_neighborhood": 1,
            "visual_channels": self.visual_channels,
            "metadata_channels": self.metadata_channels,
            "metadata_names": list(LIFTING_METADATA_NAMES),
            "hidden_dim": self.hidden_dim,
            "latent_channels": self.latent_channels,
            "zero_init_output": True,
            "time_normalization": "t_div_1000",
        }

    def forward(
        self,
        x_t: torch.Tensor,
        stock_velocity: torch.Tensor,
        t: torch.Tensor,
        visual_volume: torch.Tensor,
        metadata: torch.Tensor,
        *,
        scale: float = 1.0,
        physical_present: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected_spatial = (16, 16, 16)
        if x_t.shape != stock_velocity.shape or tuple(x_t.shape[-3:]) != expected_spatial:
            raise ValueError("x_t and stock_velocity must be aligned [B,8,16,16,16]")
        if visual_volume.shape[:2] != (x_t.shape[0], self.visual_channels):
            raise ValueError(f"invalid visual volume shape {tuple(visual_volume.shape)}")
        if metadata.shape[:2] != (x_t.shape[0], self.metadata_channels):
            raise ValueError(f"invalid metadata shape {tuple(metadata.shape)}")
        if not physical_present or float(scale) == 0.0:
            zero = torch.zeros_like(stock_velocity)
            return zero, {
                "delta_rms": zero.float().square().mean().sqrt(),
                "delta_abs_max": zero.float().abs().max(),
                "support_ratio": metadata[:, SUPPORT_METADATA_INDEX].gt(0).float().mean(),
            }
        batch = int(x_t.shape[0])
        t_channel = (t.float() / 1000.0).reshape(batch, 1, 1, 1, 1).expand(
            batch, 1, *expected_spatial
        )
        state = torch.cat((x_t.float(), stock_velocity.float(), t_channel), dim=1)
        visual = visual_volume.float().permute(0, 2, 3, 4, 1)
        visual = F.layer_norm(visual, (self.visual_channels,)).permute(
            0, 4, 1, 2, 3
        ).contiguous()
        state_hidden = self.state_projection(state)
        visual_hidden = self.visual_projection(visual)
        metadata_hidden = self.metadata_projection(metadata.float())
        interaction = torch.cat(
            (
                state_hidden,
                visual_hidden,
                state_hidden * visual_hidden,
                (state_hidden - visual_hidden).abs(),
                metadata_hidden,
            ),
            dim=1,
        )
        support_gate = metadata[:, SUPPORT_METADATA_INDEX : SUPPORT_METADATA_INDEX + 1].float()
        delta = self.output(self.fusion(interaction)) * support_gate * float(scale)
        stats = {
            "delta_rms": delta.float().square().mean().sqrt(),
            "delta_abs_max": delta.float().abs().max(),
            "support_ratio": support_gate.gt(0).float().mean(),
            "visual_rms": visual_volume.float().square().mean().sqrt(),
            "state_hidden_rms": state_hidden.float().square().mean().sqrt(),
            "visual_hidden_rms": visual_hidden.float().square().mean().sqrt(),
        }
        return delta.to(dtype=stock_velocity.dtype), stats


class LocalPoseLiftingSSFlowModel(nn.Module):
    def __init__(self, stock_flow: nn.Module, adapter: LocalPoseLiftingVelocityAdapter) -> None:
        super().__init__()
        self.stock_flow = stock_flow
        self.adapter = adapter
        self.stock_flow.eval()
        for parameter in self.stock_flow.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def stock_prediction(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.stock_flow(x_t, t, condition)

    def adapt_from_stock(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        stock_velocity: torch.Tensor,
        visual_volume: torch.Tensor,
        metadata: torch.Tensor,
        *,
        scale: float = 1.0,
        physical_present: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        delta, stats = self.adapter(
            x_t,
            stock_velocity,
            t,
            visual_volume,
            metadata,
            scale=scale,
            physical_present=physical_present,
        )
        return stock_velocity + delta, stats

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None,
        visual_volume: torch.Tensor,
        metadata: torch.Tensor,
        *,
        stock_velocity: torch.Tensor | None = None,
        corrupted_visual_volume: torch.Tensor | None = None,
        corrupted_metadata: torch.Tensor | None = None,
        scale: float = 1.0,
        physical_present: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor] | None,
    ]:
        if stock_velocity is None:
            if condition is None:
                raise ValueError("condition is required when stock_velocity is not supplied")
            stock = self.stock_prediction(x_t, t, condition)
        else:
            stock = stock_velocity
        prediction, stats = self.adapt_from_stock(
            x_t,
            t,
            stock,
            visual_volume,
            metadata,
            scale=scale,
            physical_present=physical_present,
        )
        corrupted_prediction = None
        corrupted_stats = None
        if corrupted_visual_volume is not None or corrupted_metadata is not None:
            if corrupted_visual_volume is None or corrupted_metadata is None:
                raise ValueError("corrupted visual volume and metadata must be supplied together")
            corrupted_prediction, corrupted_stats = self.adapt_from_stock(
                x_t,
                t,
                stock,
                corrupted_visual_volume,
                corrupted_metadata,
                scale=scale,
                physical_present=physical_present,
            )
        return prediction, corrupted_prediction, stock, stats, corrupted_stats
