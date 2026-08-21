from __future__ import annotations

from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from ar_ss_flow.pose_lifting import _prior_volume_features
from pose_point_depth_mv.correspondence_head import (
    CORRESPONDENCE_CHECKPOINT_VERSION,
    ViewCorrespondenceHead,
    correct_voxel_reliability_weight,
)
from pose_point_depth_mv.view_identity_lifting import (
    SPATIAL_TOLERANCE_VERSION,
    VIEW_IDENTITY_EVIDENCE_VERSION,
    VIEW_IDENTITY_GEOMETRY_NAMES,
    apply_symmetric_spatial_tolerance,
    build_view_identity_evidence,
    view_identity_schema_hash,
)
from reconvggt_ar_adapter_a.pointpose_ss_condition import lora_disabled


DIRECT_FLOW_VERSION = "pose_point_depth_mv.direct_view_identity_lora_flow.v3"
DIRECT_SPATIAL_MAPPING_VERSION = (
    "pose_point_depth_mv.xyz_volume_to_flow_patch1_tokens.v1"
)
DIRECT_EVIDENCE_BUILDER_VERSION = (
    "pose_point_depth_mv.direct_view_identity_evidence.gaussian3.v1"
)
DIRECT_CORRUPTION_MODES = (
    "pose_cyclic1",
    "depth_view_cyclic1",
    "visual_view_cyclic1",
)
DIRECT_METADATA_NAMES = (
    "active_support",
    "correspondence_score_tanh4",
    "reliability",
    "raw_reliability",
    "view_support_fraction",
    "pair_support_quality",
    "depth_reliability",
    "depth_confidence_mean",
    "depth_consistency_mean",
    "visibility_agreement",
    "prior_occupancy",
    "prior_confidence",
    "prior_distance",
    "x",
    "y",
    "z",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def names_sha256(values: Iterable[str]) -> str:
    return canonical_json_sha256(list(values))


def volume_xyz_to_flow_tokens(volume: torch.Tensor) -> torch.Tensor:
    """Map [B,C,X,Y,Z] to patch-size-1 Flow tokens in x-major order."""

    if volume.ndim != 5 or tuple(volume.shape[-3:]) != (16, 16, 16):
        raise ValueError(
            "Flow volume must be [B,C,16,16,16], got "
            f"{tuple(volume.shape)}"
        )
    return volume.flatten(2).transpose(1, 2).contiguous()


def flow_tokens_to_volume_xyz(tokens: torch.Tensor) -> torch.Tensor:
    """Inverse of volume_xyz_to_flow_tokens for 4096 patch-size-1 tokens."""

    if tokens.ndim != 3 or int(tokens.shape[1]) != 16**3:
        raise ValueError(
            "Flow tokens must be [B,4096,C], got " f"{tuple(tokens.shape)}"
        )
    return tokens.transpose(1, 2).reshape(
        int(tokens.shape[0]), int(tokens.shape[2]), 16, 16, 16
    ).contiguous()


def lifting_cache_identity(
    manifest_path: str | Path,
    *,
    rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return split identity and a path-independent lifting schema hash."""

    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected_rows = list(payload.get("samples", ())) if rows is None else list(rows)
    if not selected_rows:
        raise ValueError(f"lifting cache subset is empty: {path}")
    uids = sorted(str(row.get("uid", "")) for row in selected_rows)
    object_uids = sorted(
        {str(row.get("object_uid", row.get("uid", ""))) for row in selected_rows}
    )
    if not all(uids) or not all(object_uids):
        raise ValueError(f"lifting cache contains empty UID values: {path}")
    schema = {
        "format": payload.get("format"),
        "stock_condition_source": payload.get("stock_condition_source"),
        "lifting_feature_source": payload.get("lifting_feature_source"),
        "visual_feature_dim": int(payload.get("visual_feature_dim", 0)),
        "feature_metadata": payload.get("feature_metadata"),
        "metadata_names": payload.get("metadata_names"),
        "metadata_schema_hash": payload.get("metadata_schema_hash"),
        "config": payload.get("config"),
        "config_hash": payload.get("config_hash"),
    }
    if schema["stock_condition_source"] != "native_unmodified_reconviagen_vggt":
        raise ValueError(
            "lifting cache is missing the native unmodified ReconViaGen stock condition"
        )
    source_path_text = str(payload.get("source_cache_manifest", ""))
    source_path = Path(source_path_text)
    if source_path_text and not source_path.is_absolute():
        source_path = path.parent / source_path
    source_hash = (
        file_sha256(source_path)
        if source_path_text and source_path.is_file()
        else None
    )
    return {
        "manifest": str(path),
        "manifest_sha256": file_sha256(path),
        "cache_schema_hash": canonical_json_sha256(schema),
        "cache_config_hash": str(payload.get("config_hash", "")),
        "sample_count": len(selected_rows),
        "object_count": len(object_uids),
        "uid_hash": canonical_json_sha256(uids),
        "object_uid_hash": canonical_json_sha256(object_uids),
        "source_cache_manifest": (
            str(source_path.resolve()) if source_path_text else ""
        ),
        "source_cache_manifest_sha256": source_hash,
    }


def validate_n3_checkpoint(
    n3_report_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    report_path = Path(n3_report_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise RuntimeError(f"N3 report did not pass: {report_path}")
    matches = [
        row
        for row in report.get("per_seed", [])
        if Path(str(row.get("checkpoint", ""))).resolve() == checkpoint_path
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"correspondence checkpoint is not uniquely bound to N3: {checkpoint_path}"
        )
    row = matches[0]
    actual_hash = file_sha256(checkpoint_path)
    if actual_hash != str(row.get("checkpoint_sha256", "")):
        raise RuntimeError("correspondence checkpoint SHA-256 differs from N3")
    return {
        "n3_report": str(report_path),
        "n3_stage": str(report.get("stage", "")),
        "n3_seed": int(row["seed"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": actual_hash,
        "n3_protocol_signature": report.get("protocol_signature"),
        "n3_passed": True,
    }


def parse_csv(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("CSV value must be non-empty")
    return values


def parse_optional_csv(text: str | None) -> tuple[str, ...]:
    if text is None or not str(text).strip():
        return ()
    return tuple(item.strip() for item in str(text).split(",") if item.strip())


def load_frozen_correspondence_head(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    visual_channels: int,
) -> tuple[ViewCorrespondenceHead, dict[str, Any], dict[str, Any]]:
    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("format") != CORRESPONDENCE_CHECKPOINT_VERSION:
        raise ValueError(
            f"unexpected correspondence checkpoint format={checkpoint.get('format')!r}"
        )
    saved_args = checkpoint.get("args", {})
    head = ViewCorrespondenceHead(
        visual_channels=int(visual_channels),
        hidden_dim=int(saved_args["hidden_dim"]),
        pair_hidden_dim=int(saved_args["pair_hidden_dim"]),
        min_views=int(saved_args["min_views"]),
    ).to(device)
    head.load_state_dict(checkpoint["model_trainable_state"], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    expected_metadata = checkpoint.get("model_summary", {}).get("head")
    if expected_metadata is not None and head.metadata() != expected_metadata:
        raise ValueError("runtime correspondence head differs from checkpoint metadata")
    protocol = checkpoint.get("model_summary", {}).get("protocol", {})
    spatial_tolerance = str(
        protocol.get(
            "training_spatial_tolerance",
            saved_args.get("spatial_tolerance", "exact"),
        )
    )
    if spatial_tolerance != "gaussian3":
        raise ValueError(
            "direct Flow requires the N3 gaussian3 correspondence protocol"
        )
    runtime = {
        "checkpoint": str(path),
        "checkpoint_sha256": file_sha256(path),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "head": head.metadata(),
        "spatial_tolerance": spatial_tolerance,
        "amp_dtype": str(saved_args.get("amp_dtype", "bf16")),
        "min_views": int(saved_args["min_views"]),
        "reliability_floor": float(saved_args.get("voxel_reliability_floor", 0.10)),
        "reliability_power": float(saved_args.get("voxel_reliability_power", 1.0)),
    }
    return head, checkpoint, runtime


def _amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        from contextlib import nullcontext

        return nullcontext()
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _canonical_xyz(side: int, device: torch.device) -> torch.Tensor:
    axis = (torch.arange(side, device=device, dtype=torch.float32) + 0.5)
    axis = axis / float(side) * 2.0 - 1.0
    return torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"), dim=0
    )


@torch.no_grad()
def make_direct_evidence_bundle(
    sample: dict[str, Any],
    *,
    modes: Iterable[str],
    device: torch.device,
    correspondence_head: ViewCorrespondenceHead,
    correspondence_runtime: dict[str, Any],
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]]:
    """Build correct/corrupt branches through the exact admitted N3 protocol."""

    requested = tuple(dict.fromkeys(("correct", *tuple(modes))))
    invalid = [
        mode
        for mode in requested
        if mode not in ("correct", *DIRECT_CORRUPTION_MODES)
    ]
    if invalid:
        raise ValueError(f"unsupported direct Flow evidence modes={invalid}")

    exact = {
        mode: build_view_identity_evidence(sample, device=device, mode=mode)
        for mode in requested
    }
    exact_correct_weight = exact["correct"]["view_weight"].float()
    smoothed: dict[str, dict[str, Any]] = {}
    fixed_weight: torch.Tensor | None = None
    for mode, evidence in exact.items():
        branch, branch_weight = apply_symmetric_spatial_tolerance(
            evidence,
            fixed_correct_weight=exact_correct_weight,
            mode="gaussian3",
        )
        smoothed[mode] = branch
        if fixed_weight is None:
            fixed_weight = branch_weight
        elif not torch.equal(fixed_weight, branch_weight):
            raise RuntimeError("direct Flow branches do not share fixed support")
    assert fixed_weight is not None

    amp_dtype = str(correspondence_runtime["amp_dtype"])
    with _amp_context(device, amp_dtype):
        correct_result = correspondence_head(
            smoothed["correct"], view_weight_override=fixed_weight
        )
    reliability = correct_voxel_reliability_weight(
        smoothed["correct"],
        correct_result["active_mask"],
        min_views=int(correspondence_runtime["min_views"]),
        floor=float(correspondence_runtime["reliability_floor"]),
        power=float(correspondence_runtime["reliability_power"]),
    )

    side = 16
    occupancy, prior_confidence, prior_distance = _prior_volume_features(
        sample["prior_coords"],
        sample["prior_confidence"],
        device=device,
        volume_side=side,
    )
    xyz = _canonical_xyz(side, device).reshape(3, -1)
    active = correct_result["active_mask"].float().reshape(-1)
    shared_flat = (
        active,
        reliability["weight"].reshape(-1),
        reliability["raw_weight"].reshape(-1),
        reliability["view_support_fraction"].reshape(-1),
        reliability["pair_support_quality"].reshape(-1),
        reliability["depth_reliability"].reshape(-1),
        reliability["depth_confidence_mean"].reshape(-1),
        reliability["depth_consistency_mean"].reshape(-1),
        reliability["visibility_agreement"].reshape(-1),
        occupancy.reshape(-1),
        prior_confidence.reshape(-1),
        prior_distance.reshape(-1),
        *(xyz[index] * active for index in range(3)),
    )

    output = {}
    for mode in requested:
        if mode == "correct":
            result = correct_result
        else:
            with _amp_context(device, amp_dtype):
                result = correspondence_head(
                    smoothed[mode], view_weight_override=fixed_weight
                )
        normalized_score = torch.tanh(result["voxel_score"].float() / 4.0)
        metadata_flat = torch.stack(
            (shared_flat[0], normalized_score, *shared_flat[1:]),
            dim=0,
        )
        if int(metadata_flat.shape[0]) != len(DIRECT_METADATA_NAMES):
            raise RuntimeError("direct metadata schema construction failed")
        branch = smoothed[mode]
        visual = branch["sampled_visual"].to(dtype=torch.float16).unsqueeze(0)
        geometry = branch["geometry"].to(dtype=torch.float16).unsqueeze(0)
        weight = fixed_weight.to(dtype=torch.float16).unsqueeze(0)
        metadata = metadata_flat.reshape(
            1, len(DIRECT_METADATA_NAMES), side, side, side
        )
        active_values = correct_result["active_mask"]
        if bool(active_values.any()):
            reliability_mean = float(
                reliability["weight"][active_values].mean().item()
            )
            correspondence_score_mean = float(
                result["voxel_score"][active_values].float().mean().item()
            )
        else:
            # Empty support is valid for a few degenerate views. Keep diagnostics
            # finite; the physical branch is still gated off by active_support.
            reliability_mean = 0.0
            correspondence_score_mean = 0.0
        output[mode] = (
            visual,
            geometry,
            weight,
            metadata,
            {
                "mode": mode,
                "views": int(branch["views"]),
                "active_ratio": float(active.mean().item()),
                "reliability_mean": reliability_mean,
                "correspondence_score_mean": correspondence_score_mean,
                "prior_point_count": int(sample["prior_coords"].shape[0]),
                "fixed_correct_support": True,
                "spatial_tolerance": "gaussian3",
            },
        )
    return output


def null_evidence_like(
    evidence: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.zeros_like(value) for value in evidence[:4])  # type: ignore[return-value]


class DirectViewTokenEncoder(nn.Module):
    """Fuse per-view visual/geometry identity into exact 16^3 Flow tokens."""

    def __init__(
        self,
        *,
        visual_channels: int,
        flow_channels: int,
        hidden_dim: int = 128,
        geometry_channels: int = len(VIEW_IDENTITY_GEOMETRY_NAMES),
        metadata_channels: int = len(DIRECT_METADATA_NAMES),
    ) -> None:
        super().__init__()
        self.visual_channels = int(visual_channels)
        self.geometry_channels = int(geometry_channels)
        self.metadata_channels = int(metadata_channels)
        self.flow_channels = int(flow_channels)
        self.hidden_dim = int(hidden_dim)
        if min(
            self.visual_channels,
            self.geometry_channels,
            self.metadata_channels,
            self.flow_channels,
            self.hidden_dim,
        ) <= 0:
            raise ValueError("direct Flow encoder dimensions must be positive")
        if self.metadata_channels != len(DIRECT_METADATA_NAMES):
            raise ValueError("direct Flow metadata schema mismatch")
        self.active_index = DIRECT_METADATA_NAMES.index("active_support")
        self.prior_index = DIRECT_METADATA_NAMES.index("prior_occupancy")

        self.visual_norm = nn.LayerNorm(self.visual_channels)
        self.visual_projection = nn.Linear(
            self.visual_channels, self.hidden_dim, bias=False
        )
        self.geometry_norm = nn.LayerNorm(self.geometry_channels)
        self.geometry_projection = nn.Linear(
            self.geometry_channels, self.hidden_dim, bias=False
        )
        self.view_fusion = nn.Sequential(
            nn.LayerNorm(4 * self.hidden_dim),
            nn.Linear(4 * self.hidden_dim, 2 * self.hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(3 * self.hidden_dim),
            nn.Linear(3 * self.hidden_dim, self.hidden_dim, bias=False),
            nn.SiLU(),
        )
        self.metadata_projection = nn.Conv3d(
            self.metadata_channels, self.hidden_dim, 1, bias=False
        )
        self.pair_query = nn.Conv3d(
            2 * self.hidden_dim, self.hidden_dim, 1, bias=False
        )
        self.local_fusion = nn.Sequential(
            nn.Conv3d(
                6 * self.hidden_dim,
                2 * self.hidden_dim,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(self._group_count(2 * self.hidden_dim), 2 * self.hidden_dim),
            nn.SiLU(),
            nn.Conv3d(
                2 * self.hidden_dim,
                self.hidden_dim,
                3,
                padding=1,
                bias=False,
            ),
            nn.SiLU(),
        )
        self.output = nn.Conv3d(
            self.hidden_dim, self.flow_channels, 1, bias=False
        )
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def _group_count(channels: int) -> int:
        groups = min(16, int(channels))
        while int(channels) % groups:
            groups -= 1
        return groups

    def metadata(self) -> dict[str, Any]:
        return {
            "version": DIRECT_FLOW_VERSION,
            "type": type(self).__name__,
            "input_grid": [16, 16, 16],
            "flow_token_grid": [16, 16, 16],
            "spatial_mapping": "x_major_same_voxel_flow_patch_size_1",
            "spatial_mapping_version": DIRECT_SPATIAL_MAPPING_VERSION,
            "view_identity_preserved_until": "same_voxel_pair_fusion",
            "visual_channels": self.visual_channels,
            "geometry_channels": self.geometry_channels,
            "geometry_names": list(VIEW_IDENTITY_GEOMETRY_NAMES),
            "geometry_names_sha256": names_sha256(
                VIEW_IDENTITY_GEOMETRY_NAMES
            ),
            "metadata_channels": self.metadata_channels,
            "metadata_names": list(DIRECT_METADATA_NAMES),
            "metadata_names_sha256": names_sha256(DIRECT_METADATA_NAMES),
            "evidence_builder_version": DIRECT_EVIDENCE_BUILDER_VERSION,
            "view_identity_evidence_version": VIEW_IDENTITY_EVIDENCE_VERSION,
            "view_identity_schema_hash": view_identity_schema_hash(),
            "spatial_tolerance_version": SPATIAL_TOLERANCE_VERSION,
            "hidden_dim": self.hidden_dim,
            "flow_channels": self.flow_channels,
            "local_context": "3x3x3_on_flow_token_grid",
            "zero_init_output": True,
            "null_policy": "empty_fixed_support_disables_physical_and_lora",
        }

    def evidence_present(
        self, metadata: torch.Tensor, view_weight: torch.Tensor
    ) -> bool:
        if int(metadata.shape[0]) != 1:
            raise ValueError("direct Flow currently requires per-rank batch size 1")
        active = metadata[:, self.active_index].gt(0).any()
        prior = metadata[:, self.prior_index].gt(0).any()
        views = view_weight.gt(0).any()
        return bool((active | prior | views).item())

    @staticmethod
    def _prepare_per_view(
        values: torch.Tensor, view_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 4 or view_weight.ndim != 3:
            raise ValueError("per-view values/weight must be [B,V,N,C]/[B,V,N]")
        batch, views, voxel_count, channels = map(int, values.shape)
        if voxel_count != 16**3 or tuple(view_weight.shape) != (
            batch,
            views,
            voxel_count,
        ):
            raise ValueError("per-view evidence shape mismatch")
        return values.float(), view_weight.float()

    def forward(
        self,
        view_visual: torch.Tensor,
        view_geometry: torch.Tensor,
        view_weight: torch.Tensor,
        metadata: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if view_visual.ndim != 4 or view_geometry.ndim != 4:
            raise ValueError("view evidence must be [B,V,4096,C]")
        batch, views, voxels, visual_channels = map(int, view_visual.shape)
        if views < 2 or voxels != 16**3 or visual_channels != self.visual_channels:
            raise ValueError(f"invalid view visual shape={tuple(view_visual.shape)}")
        if tuple(view_geometry.shape) != (
            batch,
            views,
            voxels,
            self.geometry_channels,
        ):
            raise ValueError("view geometry schema mismatch")
        if tuple(metadata.shape) != (
            batch,
            self.metadata_channels,
            16,
            16,
            16,
        ):
            raise ValueError("direct metadata shape mismatch")

        visual16, weight16 = self._prepare_per_view(view_visual, view_weight)
        geometry16, geometry_weight16 = self._prepare_per_view(
            view_geometry, view_weight
        )
        if not torch.equal(weight16, geometry_weight16):
            raise RuntimeError("visual and geometry support differs")
        visual_hidden = self.visual_projection(self.visual_norm(visual16))
        geometry_hidden = self.geometry_projection(self.geometry_norm(geometry16))
        view_hidden = self.view_fusion(
            torch.cat(
                (
                    visual_hidden,
                    geometry_hidden,
                    visual_hidden * geometry_hidden,
                    (visual_hidden - geometry_hidden).abs(),
                ),
                dim=-1,
            )
        )
        valid_view = weight16.gt(1.0e-6)
        normalized_view_weight = weight16 / weight16.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        view_mean = (
            view_hidden * normalized_view_weight[..., None]
        ).sum(dim=1)

        pair_features = []
        pair_weights = []
        for first, second in combinations(range(views), 2):
            pair_features.append(
                self.pair_projection(
                    torch.cat(
                        (
                            0.5 * (view_hidden[:, first] + view_hidden[:, second]),
                            view_hidden[:, first] * view_hidden[:, second],
                            (view_hidden[:, first] - view_hidden[:, second]).abs(),
                        ),
                        dim=-1,
                    )
                )
            )
            pair_weights.append(
                torch.minimum(weight16[:, first], weight16[:, second])
            )
        pair_hidden = torch.stack(pair_features, dim=1)
        pair_weight = torch.stack(pair_weights, dim=1)
        pair_valid = pair_weight.gt(1.0e-6)

        metadata_hidden = self.metadata_projection(metadata.float())
        view_volume = flow_tokens_to_volume_xyz(view_mean)
        query = self.pair_query(
            torch.cat((view_volume, metadata_hidden), dim=1)
        )
        query = volume_xyz_to_flow_tokens(query)
        scores = (pair_hidden * query[:, None]).sum(dim=-1) / math.sqrt(
            float(self.hidden_dim)
        )
        scores = scores.masked_fill(~pair_valid, -1.0e4)
        attention = torch.softmax(scores, dim=1) * pair_valid.float()
        attention = attention / attention.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        attended = (pair_hidden * attention[..., None]).sum(dim=1)
        attended_volume = flow_tokens_to_volume_xyz(attended)
        interaction = torch.cat(
            (
                view_volume,
                metadata_hidden,
                attended_volume,
                view_volume * attended_volume,
                (view_volume - attended_volume).abs(),
                metadata_hidden * attended_volume,
            ),
            dim=1,
        )
        active16 = metadata[
            :, self.active_index : self.active_index + 1
        ].float()
        prior16 = metadata[
            :, self.prior_index : self.prior_index + 1
        ].float()
        evidence_gate = torch.maximum(active16, prior16).clamp(0.0, 1.0)
        token_volume = self.output(self.local_fusion(interaction)) * evidence_gate
        tokens = volume_xyz_to_flow_tokens(token_volume)

        entropy = -(attention.clamp_min(1.0e-8).log() * attention).sum(dim=1)
        valid_voxel = pair_valid.any(dim=1)
        entropy_mean = (
            entropy[valid_voxel].mean()
            if bool(valid_voxel.any().item())
            else entropy.new_zeros(())
        )
        return tokens, {
            "physical_token_rms": tokens.float().square().mean().sqrt(),
            "physical_token_abs_max": tokens.float().abs().amax(),
            "evidence_gate_ratio": evidence_gate.gt(0).float().mean(),
            "valid_view_ratio": valid_view.float().mean(),
            "pair_valid_ratio": pair_valid.float().mean(),
            "pair_attention_entropy": entropy_mean,
        }


class DirectPhysicalFlowModel(nn.Module):
    """Physical-token SS Flow with LoRA and an exact native-stock bypass."""

    def __init__(self, lora_flow: nn.Module, physical_encoder: DirectViewTokenEncoder) -> None:
        super().__init__()
        self.flow = lora_flow
        self.physical_encoder = physical_encoder
        core = self.flow_core
        if (
            int(core.resolution) != 16
            or int(core.in_channels) != 8
            or int(core.out_channels) != 8
            or int(core.patch_size) != 1
        ):
            raise ValueError("direct Flow requires the stock 8x16^3 patch-size-1 schema")
        if int(core.model_channels) != int(physical_encoder.flow_channels):
            raise ValueError("physical token channels differ from Flow model channels")

    @property
    def flow_core(self) -> nn.Module:
        base_model = getattr(self.flow, "base_model", None)
        core = getattr(base_model, "model", None)
        return core if isinstance(core, nn.Module) else self.flow

    def _core_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | list[torch.Tensor],
        physical_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        from trellis.modules.spatial import patchify, unpatchify

        core = self.flow_core
        expected = [
            x.shape[0],
            core.in_channels,
            core.resolution,
            core.resolution,
            core.resolution,
        ]
        if list(x.shape) != expected:
            raise ValueError(f"Flow input shape {list(x.shape)} != {expected}")
        h = volume_xyz_to_flow_tokens(patchify(x, core.patch_size))
        h = core.input_layer(h)
        h = h + core.pos_emb[None]
        t_emb = core.t_embedder(t)
        if core.share_mod:
            t_emb = core.adaLN_modulation(t_emb)
        t_emb = t_emb.type(core.dtype)
        h = h.type(core.dtype)
        if physical_tokens is not None:
            if physical_tokens.shape != h.shape:
                raise ValueError(
                    f"physical tokens {tuple(physical_tokens.shape)} != "
                    f"Flow tokens {tuple(h.shape)}"
                )
            h = h + physical_tokens.to(dtype=h.dtype)
        contexts = cond if isinstance(cond, list) else [cond]
        for context_value in contexts:
            context = context_value.type(core.dtype)
            for block in core.blocks:
                h = block(h, t_emb, context)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = core.out_layer(h)
        h = flow_tokens_to_volume_xyz(h)
        return unpatchify(h, core.patch_size).contiguous()

    @torch.no_grad()
    def stock_prediction(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        with lora_disabled(self.flow):
            return self._core_forward(x_t, t, condition, None)

    def conditioned_prediction(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        view_visual: torch.Tensor,
        view_geometry: torch.Tensor,
        view_weight: torch.Tensor,
        metadata: torch.Tensor,
        *,
        stock_velocity: torch.Tensor | None = None,
        physical_scale: float = 1.0,
        physical_present: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stock = (
            self.stock_prediction(x_t, t, condition)
            if stock_velocity is None
            else stock_velocity
        )
        present = bool(physical_present) and float(physical_scale) != 0.0
        present = present and self.physical_encoder.evidence_present(
            metadata, view_weight
        )
        if not present:
            zero = stock.new_zeros((), dtype=torch.float32)
            return stock, {
                "physical_token_rms": zero,
                "physical_token_abs_max": zero,
                "evidence_gate_ratio": zero,
                "valid_view_ratio": zero,
                "pair_valid_ratio": zero,
                "pair_attention_entropy": zero,
                "flow_delta_rms": zero,
                "flow_delta_abs_max": zero,
                "physical_present": zero,
            }
        tokens, stats = self.physical_encoder(
            view_visual, view_geometry, view_weight, metadata
        )
        prediction = self._core_forward(
            x_t, t, condition, tokens * float(physical_scale)
        )
        delta = prediction.float() - stock.float()
        stats = dict(stats)
        stats.update(
            {
                "flow_delta_rms": delta.square().mean().sqrt(),
                "flow_delta_abs_max": delta.abs().amax(),
                "physical_present": delta.new_tensor(1.0),
            }
        )
        return prediction, stats

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        view_visual: torch.Tensor,
        view_geometry: torch.Tensor,
        view_weight: torch.Tensor,
        metadata: torch.Tensor,
        *,
        stock_velocity: torch.Tensor | None = None,
        wrong_view_visual: torch.Tensor | None = None,
        wrong_view_geometry: torch.Tensor | None = None,
        wrong_view_weight: torch.Tensor | None = None,
        wrong_metadata: torch.Tensor | None = None,
        physical_scale: float = 1.0,
        physical_present: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor] | None,
    ]:
        stock = (
            self.stock_prediction(x_t, t, condition)
            if stock_velocity is None
            else stock_velocity
        )
        prediction, stats = self.conditioned_prediction(
            x_t,
            t,
            condition,
            view_visual,
            view_geometry,
            view_weight,
            metadata,
            stock_velocity=stock,
            physical_scale=physical_scale,
            physical_present=physical_present,
        )
        wrong_values = (
            wrong_view_visual,
            wrong_view_geometry,
            wrong_view_weight,
            wrong_metadata,
        )
        wrong_prediction = None
        wrong_stats = None
        if any(value is not None for value in wrong_values):
            if not all(torch.is_tensor(value) for value in wrong_values):
                raise ValueError("all wrong-evidence tensors must be provided together")
            wrong_prediction, wrong_stats = self.conditioned_prediction(
                x_t,
                t,
                condition,
                wrong_view_visual,
                wrong_view_geometry,
                wrong_view_weight,
                wrong_metadata,
                stock_velocity=stock,
                physical_scale=physical_scale,
                physical_present=physical_present,
            )
        return prediction, wrong_prediction, stock, stats, wrong_stats


class NativeStockFlow(nn.Module):
    def __init__(self, model: DirectPhysicalFlowModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.model.stock_prediction(x_t, t, condition)


class PositivePhysicalRolloutFlow(nn.Module):
    """Enable physical tokens and LoRA only on the positive CFG branch."""

    def __init__(
        self,
        model: DirectPhysicalFlowModel,
        positive_condition: torch.Tensor,
        evidence: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        physical_scale: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.positive_condition = positive_condition
        self.evidence = evidence
        self.physical_scale = float(physical_scale)
        self.positive_calls = 0
        self.negative_calls = 0

    def _is_positive(self, condition: torch.Tensor) -> bool:
        return condition is self.positive_condition or (
            condition.shape == self.positive_condition.shape
            and condition.data_ptr() == self.positive_condition.data_ptr()
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if not self._is_positive(condition):
            self.negative_calls += 1
            return self.model.stock_prediction(x_t, t, condition)
        self.positive_calls += 1
        prediction, _ = self.model.conditioned_prediction(
            x_t,
            t,
            condition,
            *self.evidence,
            physical_scale=self.physical_scale,
        )
        return prediction
