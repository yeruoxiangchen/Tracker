#!/usr/bin/env python3
"""Run the frozen synthetic-only reviewed1k no-VGGT SS/SLat deployment."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv import infer_omni_real_native_v2 as _v2_infer
from pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs import (
    MANIFEST_FORMAT as MODEL_INPUT_MANIFEST_FORMAT,
    OBJECT_FORMAT as MODEL_INPUT_OBJECT_FORMAT,
)
from pose_point_depth_mv.native_slat_genrecon_no_vggt import (
    NATIVE_SLAT_NO_VGGT_VERSION,
    NO_VGGT_SLAT_CONTRACT,
    build_native_slat_no_vggt_components,
    validate_native_slat_no_vggt_checkpoint,
)
from pose_point_depth_mv.native_slat_genrecon_v2 import load_stock_slat_freeze
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    NATIVE_SS_NO_VGGT_VERSION,
    NO_VGGT_MODEL_CONTRACT,
    build_native_ss_no_vggt_components,
    validate_native_ss_no_vggt_checkpoint,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
)


REPORT_FORMAT = "pose_point_depth_mv.omni_real_native_no_vggt_synthetic_inference.v1"
MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_native_no_vggt_synthetic_inference_manifest.v1"
)
EXPECTED_OBJECT_COUNT = 868
EXPECTED_SAMPLE_COUNT = 1417


def _argument(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        prefix = f"{name}="
        return next(
            (value[len(prefix) :] for value in sys.argv if value.startswith(prefix)),
            None,
        )


def _identity_summary(checkpoint: dict[str, Any]) -> dict[str, Any]:
    identity = dict(checkpoint.get("data_identity", {}))
    return {
        name: identity.get(name)
        for name in (
            "manifest",
            "manifest_sha256",
            "cache_manifest",
            "cache_manifest_sha256",
            "config_hash",
            "sample_count",
            "object_count",
        )
        if identity.get(name) is not None
    }


def validate_synthetic_deployment(
    *,
    pretrained: str,
    ss_path: Path,
    slat_path: Path,
    stock_freeze_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate synthetic lineage directly, without a real-Full migration contract."""

    ss_checkpoint = torch.load(ss_path, map_location="cpu")
    slat_checkpoint = torch.load(slat_path, map_location="cpu")
    if ss_checkpoint.get("format") != NATIVE_SS_NO_VGGT_VERSION:
        raise ValueError("synthetic profile requires a trained no-VGGT Native SS")
    if slat_checkpoint.get("format") != NATIVE_SLAT_NO_VGGT_VERSION:
        raise ValueError("synthetic profile requires a trained no-VGGT Native SLat")
    for stage, checkpoint in (("ss", ss_checkpoint), ("slat", slat_checkpoint)):
        migration = checkpoint.get("model_summary", {}).get("migration_contract")
        if migration not in (None, {}):
            raise RuntimeError(
                f"synthetic-only {stage} unexpectedly declares a real migration contract"
            )
        identity = dict(checkpoint.get("data_identity", {}))
        if int(identity.get("object_count", -1)) != EXPECTED_OBJECT_COUNT:
            raise RuntimeError(
                f"synthetic-only {stage} object_count is not {EXPECTED_OBJECT_COUNT}"
            )
        if int(identity.get("sample_count", -1)) != EXPECTED_SAMPLE_COUNT:
            raise RuntimeError(
                f"synthetic-only {stage} sample_count is not {EXPECTED_SAMPLE_COUNT}"
            )
    validate_native_ss_no_vggt_checkpoint(
        ss_checkpoint, pretrained=pretrained, allow_v2_parent=False
    )
    upstream = dict(
        slat_checkpoint.get("model_summary", {}).get("upstream_native_ss", {})
    )
    ss_sha256 = sha256_file(ss_path)
    if upstream.get("checkpoint_sha256") != ss_sha256:
        raise RuntimeError("synthetic-only SLat is not bound to the requested SS")
    stock_freeze = load_stock_slat_freeze(stock_freeze_path)
    validate_native_slat_no_vggt_checkpoint(
        slat_checkpoint,
        pretrained=pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=upstream,
        allow_v2_parent=False,
    )
    lineage = {
        "profile": "reviewed1k_synthetic868_no_real_video_v1",
        "object_count": EXPECTED_OBJECT_COUNT,
        "sequence_count": EXPECTED_SAMPLE_COUNT,
        "observation_domain": "Blender synthetic RGB/mask/K/T; no Omni real video",
        "ss_checkpoint": str(ss_path),
        "ss_checkpoint_sha256": ss_sha256,
        "slat_checkpoint": str(slat_path),
        "slat_checkpoint_sha256": sha256_file(slat_path),
        "ss_data_identity": _identity_summary(ss_checkpoint),
        "slat_data_identity": _identity_summary(slat_checkpoint),
        "real_full_migration_contract_consumed": False,
    }
    return upstream, lineage


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        _v2_infer.__doc__ = __doc__
        _v2_infer.main()
        return
    ss_value = _argument("--native_ss_checkpoint")
    slat_value = _argument("--native_slat_checkpoint")
    stock_value = _argument("--stock_slat_freeze")
    output_value = _argument("--output_dir")
    pretrained = _argument("--pretrained") or "Stable-X/trellis-vggt-v0-2"
    if not all((ss_value, slat_value, stock_value, output_value)):
        raise ValueError("SS, SLat, stock freeze, and output_dir are required")
    ss_path = Path(str(ss_value)).expanduser().resolve()
    slat_path = Path(str(slat_value)).expanduser().resolve()
    stock_path = Path(str(stock_value)).expanduser().resolve()
    _upstream, lineage = validate_synthetic_deployment(
        pretrained=pretrained,
        ss_path=ss_path,
        slat_path=slat_path,
        stock_freeze_path=stock_path,
    )

    _v2_infer.MODEL_INPUT_MANIFEST_FORMAT = MODEL_INPUT_MANIFEST_FORMAT
    _v2_infer.MODEL_INPUT_OBJECT_FORMAT = MODEL_INPUT_OBJECT_FORMAT
    _v2_infer.REPORT_FORMAT = REPORT_FORMAT
    _v2_infer.MANIFEST_FORMAT = MANIFEST_FORMAT
    _v2_infer.__doc__ = __doc__
    _v2_infer.validate_native_ss_genrecon_checkpoint = (
        validate_native_ss_no_vggt_checkpoint
    )
    _v2_infer.build_native_ss_genrecon_components = (
        build_native_ss_no_vggt_components
    )
    _v2_infer.validate_native_slat_genrecon_v2_checkpoint = (
        validate_native_slat_no_vggt_checkpoint
    )
    _v2_infer.build_native_slat_genrecon_v2_components = (
        build_native_slat_no_vggt_components
    )

    original_run_slat = _v2_infer._run_slat

    def run_slat_synthetic(**kwargs: Any):
        reports = original_run_slat(**kwargs)
        for report in reports:
            report.update(
                {
                    "format": REPORT_FORMAT,
                    "method": "native_no_vggt_synthetic868",
                    "ss_input_context_contract": NO_VGGT_MODEL_CONTRACT,
                    "slat_input_context_contract": NO_VGGT_SLAT_CONTRACT,
                    "synthetic_training_lineage": lineage,
                    "vggt_model_loaded": False,
                    "vggt_model_executed": False,
                }
            )
            result_path = Path(report["mesh"]).parent / "result.json"
            atomic_json(result_path, report)
        return reports

    _v2_infer._run_slat = run_slat_synthetic
    _v2_infer.main()

    manifest_path = Path(str(output_value)).expanduser().resolve() / "inference_manifest.json"
    manifest = load_json(manifest_path)
    manifest.update(
        {
            "format": MANIFEST_FORMAT,
            "method": "native_no_vggt_synthetic868",
            "ss_input_context_contract": NO_VGGT_MODEL_CONTRACT,
            "slat_input_context_contract": NO_VGGT_SLAT_CONTRACT,
            "synthetic_training_lineage": lineage,
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
        }
    )
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
