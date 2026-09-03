"""Frozen assets and scientific identity of the paper's current 30K endpoint."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


# Existing checkpoints and reports intentionally retain their original format
# identifiers. This is artifact compatibility metadata, not a package import.
LEGACY_ARTIFACT_NAMESPACE = "pose_point_depth_mv"
PRETRAINED = "Stable-X/trellis-vggt-v0-2"
CHECKPOINT_STEP = 30_000


@dataclass(frozen=True)
class FrozenAsset:
    path: Path
    sha256: str
    size: int

    def validate(self, *, full_hash: bool) -> dict[str, Any]:
        resolved = self.path.expanduser().resolve(strict=True)
        observed_size = resolved.stat().st_size
        if observed_size != self.size:
            raise RuntimeError(
                f"asset size differs: path={resolved} observed={observed_size} "
                f"expected={self.size}"
            )
        result: dict[str, Any] = {
            "path": str(resolved),
            "size": observed_size,
            "expected_sha256": self.sha256,
        }
        if full_hash:
            digest = hashlib.sha256()
            with resolved.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            observed_hash = digest.hexdigest()
            if observed_hash != self.sha256:
                raise RuntimeError(
                    f"asset SHA256 differs: path={resolved} "
                    f"observed={observed_hash} expected={self.sha256}"
                )
            result["sha256"] = observed_hash
        return result


SS30K_REPORT = FrozenAsset(
    Path(
        "/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/"
        "ss30k_dev64_aggregate/report.json"
    ),
    "4dd4badcf1d8874bc0da5d707f4fde5bd1a6962e664598a6010f596247982d1f",
    41_812,
)
SS30K_CHECKPOINT = FrozenAsset(
    Path(
        "/data/zjr/proobjaverse_official_30k_checkpoint_archives/"
        "ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/"
        "ss/checkpoints/step_030000.pt"
    ),
    "042a1b5467b05975584aeb571dec6ffaed5096edcc6abe4aa88600c9c9506b7f",
    588_840_235,
)
SLAT30K_CHECKPOINT = FrozenAsset(
    Path(
        "/data/zjr/proobjaverse_official_30k_checkpoint_archives/"
        "ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/"
        "slat/checkpoints/step_030000.pt"
    ),
    "da8d058bbb1a917a5b91cd338d60f7d7ec15d7a5a211c86250ed514d8c0a0371",
    591_783_443,
)
ABC_R_BRIDGE = FrozenAsset(
    Path(
        "/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/"
        "abc_r_dev64_aggregate/report.json"
    ),
    "683479c5fadf34975354135a700cd69cc1897f5a97cfb982695fbf8f286a348b",
    382_472,
)
STOCK_SLAT_FREEZE = FrozenAsset(
    Path(
        "/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/"
        "stock_slat_freeze_v2.json"
    ),
    "131f9612c736a57cf41ed2161ee102ce8bcab3b9f39fd1a2473ce7919b290bc6",
    1_841,
)
FROZEN_SELECTION = FrozenAsset(
    Path(
        "/data/zjr/ProObjaverse-300K-ReconViaGen-30K-state/combined_audit/"
        "combined_selection_30k.json"
    ),
    "8125521278a97c6120a6246fe47eb0872fb905e0fb631a0c0189116a19b53f48",
    18_581_874,
)
SOURCE_AUDIT = FrozenAsset(
    Path(
        "/data/zjr/ProObjaverse-300K-ReconViaGen-30K-state/combined_audit/"
        "audit_report.json"
    ),
    "ad1947dd37fc89059fe57019dc035d78c39c75f5bcafddae42ffa700224dc364",
    19_012,
)

DEPLOYMENT_ASSETS = {
    "native_ss_report": SS30K_REPORT,
    "native_ss_checkpoint": SS30K_CHECKPOINT,
    "native_slat_checkpoint": SLAT30K_CHECKPOINT,
    "cross_deployment_bridge": ABC_R_BRIDGE,
    "stock_slat_freeze": STOCK_SLAT_FREEZE,
}
TRAINING_IDENTITY_ASSETS = {
    "frozen_selection": FROZEN_SELECTION,
    "source_audit": SOURCE_AUDIT,
}


def validate_frozen_assets(*, full_hash: bool = False) -> dict[str, Any]:
    """Validate files and deployment bindings used by the paper model."""

    assets = {**DEPLOYMENT_ASSETS, **TRAINING_IDENTITY_ASSETS}
    result = {
        label: asset.validate(full_hash=full_hash)
        for label, asset in assets.items()
    }
    ss_payload = json.loads(SS30K_REPORT.path.read_text(encoding="utf-8"))
    deployment = dict(ss_payload.get("deployment") or {})
    expected_ss = {
        "checkpoint_sha256": SS30K_CHECKPOINT.sha256,
        "checkpoint_step": CHECKPOINT_STEP,
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
        or int(bridge_slat.get("checkpoint_step", -1)) != CHECKPOINT_STEP
        or bridge_slat.get("weights") != "ema"
    ):
        raise RuntimeError("frozen A/B/C/R SS30K+SLat30K binding differs")
    result["endpoint"] = {
        "native_ss": "no-VGGT Native-SS step30000 EMA",
        "native_slat": "no-VGGT Native-SLat v2 step30000 EMA",
        "decoder": "frozen Stock Mesh decoder",
        "usable_training_objects": 29_861,
        "checkpoint_step": CHECKPOINT_STEP,
    }
    return result


__all__ = [
    "ABC_R_BRIDGE",
    "CHECKPOINT_STEP",
    "DEPLOYMENT_ASSETS",
    "FROZEN_SELECTION",
    "PRETRAINED",
    "SLAT30K_CHECKPOINT",
    "SOURCE_AUDIT",
    "SS30K_CHECKPOINT",
    "SS30K_REPORT",
    "STOCK_SLAT_FREEZE",
    "TRAINING_IDENTITY_ASSETS",
    "validate_frozen_assets",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="hash the two roughly 0.6 GB checkpoints in addition to metadata checks",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate_frozen_assets(full_hash=args.full_hash),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
