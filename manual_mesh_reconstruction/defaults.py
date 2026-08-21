"""Frozen default deployment assets for manual no-VGGT SS30K+SLat30K tests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from manual_mesh_reconstruction.common import sha256_file


@dataclass(frozen=True)
class BoundAsset:
    path: Path
    sha256: str

    def validate(self, label: str) -> Path:
        resolved = self.path.expanduser().resolve(strict=True)
        observed = sha256_file(resolved)
        if observed != self.sha256:
            raise RuntimeError(
                f"{label} SHA256 differs: observed={observed} expected={self.sha256}"
            )
        return resolved


SS30K_REPORT = BoundAsset(
    Path(
        "/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/"
        "ss30k_dev64_aggregate/report.json"
    ),
    "4dd4badcf1d8874bc0da5d707f4fde5bd1a6962e664598a6010f596247982d1f",
)
SS30K_CHECKPOINT = BoundAsset(
    Path(
        "/data/zjr/proobjaverse_official_30k_checkpoint_archives/"
        "ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/"
        "ss/checkpoints/step_030000.pt"
    ),
    "042a1b5467b05975584aeb571dec6ffaed5096edcc6abe4aa88600c9c9506b7f",
)
SLAT30K_CHECKPOINT = BoundAsset(
    Path(
        "/data/zjr/proobjaverse_official_30k_checkpoint_archives/"
        "ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/"
        "slat/checkpoints/step_030000.pt"
    ),
    "da8d058bbb1a917a5b91cd338d60f7d7ec15d7a5a211c86250ed514d8c0a0371",
)
ABC_R_BRIDGE = BoundAsset(
    Path(
        "/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/"
        "abc_r_dev64_aggregate/report.json"
    ),
    "683479c5fadf34975354135a700cd69cc1897f5a97cfb982695fbf8f286a348b",
)
STOCK_SLAT_FREEZE = BoundAsset(
    Path(
        "/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/"
        "stock_slat_freeze_v2.json"
    ),
    "131f9612c736a57cf41ed2161ee102ce8bcab3b9f39fd1a2473ce7919b290bc6",
)

SLAT_STEP = 30_000
PRETRAINED = "Stable-X/trellis-vggt-v0-2"


def validate_frozen_assets() -> dict[str, dict[str, str]]:
    assets = {
        "native_ss_report": SS30K_REPORT,
        "native_ss_checkpoint": SS30K_CHECKPOINT,
        "native_slat_checkpoint": SLAT30K_CHECKPOINT,
        "cross_deployment_bridge": ABC_R_BRIDGE,
        "stock_slat_freeze": STOCK_SLAT_FREEZE,
    }
    result: dict[str, dict[str, str]] = {}
    for label, asset in assets.items():
        result[label] = {
            "path": str(asset.validate(label)),
            "sha256": asset.sha256,
        }
    ss_payload = json.loads(SS30K_REPORT.path.read_text(encoding="utf-8"))
    deployment = dict(ss_payload.get("deployment") or {})
    expected_ss = {
        "checkpoint_sha256": SS30K_CHECKPOINT.sha256,
        "checkpoint_step": 30_000,
        "weights": "ema",
        "cfg_strength": 5.0,
        "steps": 25,
        "cfg_interval": [0.5, 1.0],
        "amp_dtype": "bf16",
    }
    if ss_payload.get("passed") is not True or any(
        deployment.get(key) != value for key, value in expected_ss.items()
    ):
        raise RuntimeError("frozen SS30K deployment binding differs")
    bridge = json.loads(ABC_R_BRIDGE.path.read_text(encoding="utf-8"))
    bridge_ss = dict(bridge.get("native_ss_binding") or {})
    bridge_slat = dict(bridge.get("native_slat_binding") or {})
    if (
        bridge.get("passed") is not True
        or bridge.get("runtime_integrity_passed") is not True
        or bridge_ss.get("report_sha256") != SS30K_REPORT.sha256
        or bridge_ss.get("checkpoint_sha256") != SS30K_CHECKPOINT.sha256
        or bridge_slat.get("checkpoint_sha256") != SLAT30K_CHECKPOINT.sha256
        or int(bridge_slat.get("checkpoint_step", -1)) != SLAT_STEP
        or bridge_slat.get("weights") != "ema"
    ):
        raise RuntimeError("frozen A/B/C/R SS30K+SLat30K binding differs")
    return result
