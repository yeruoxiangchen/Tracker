#!/usr/bin/env python3
"""Run the contracted mixed-domain no-VGGT SS and SLat deployment."""

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
from pose_point_depth_mv.real_full_no_vggt_migration import (
    load_migration_contract,
    migration_summary,
    validate_destination_migration,
)


REPORT_FORMAT = "pose_point_depth_mv.omni_real_native_no_vggt_mixed_inference.v1"
MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_native_no_vggt_mixed_inference_manifest.v1"
)


def _argument(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        prefix = f"{name}="
        return next(
            (value[len(prefix) :] for value in sys.argv if value.startswith(prefix)),
            None,
        )


def _pop_argument(name: str) -> str:
    prefix = f"{name}="
    for index, value in enumerate(list(sys.argv)):
        if value.startswith(prefix):
            del sys.argv[index]
            return value[len(prefix) :]
        if value == name:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{name} requires a value")
            result = sys.argv[index + 1]
            del sys.argv[index : index + 2]
            return result
    raise ValueError(f"{name} is required")


def _validate_deployment(
    *,
    pretrained: str,
    ss_path: Path,
    slat_path: Path,
    stock_freeze_path: Path,
    ss_contract_path: str,
    slat_contract_path: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ss_contract = load_migration_contract(ss_contract_path, stage="ss")
    slat_contract = load_migration_contract(slat_contract_path, stage="slat")
    ss_checkpoint = torch.load(ss_path, map_location="cpu")
    slat_checkpoint = torch.load(slat_path, map_location="cpu")
    if ss_checkpoint.get("format") != NATIVE_SS_NO_VGGT_VERSION:
        raise ValueError("production inference requires a trained mixed no-VGGT SS")
    if slat_checkpoint.get("format") != NATIVE_SLAT_NO_VGGT_VERSION:
        raise ValueError("production inference requires a trained mixed no-VGGT SLat")
    validate_native_ss_no_vggt_checkpoint(
        ss_checkpoint, pretrained=pretrained, allow_v2_parent=False
    )
    validate_destination_migration(ss_checkpoint, ss_contract)
    upstream = dict(slat_checkpoint.get("model_summary", {}).get("upstream_native_ss", {}))
    if upstream.get("checkpoint_sha256") != sha256_file(ss_path):
        raise RuntimeError("mixed no-VGGT SLat is not bound to the requested SS")
    stock_freeze = load_stock_slat_freeze(stock_freeze_path)
    validate_native_slat_no_vggt_checkpoint(
        slat_checkpoint,
        pretrained=pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=upstream,
        allow_v2_parent=False,
    )
    validate_destination_migration(slat_checkpoint, slat_contract)
    return ss_contract, slat_contract, upstream


def main() -> None:
    help_requested = "--help" in sys.argv or "-h" in sys.argv
    if help_requested:
        _v2_infer.__doc__ = __doc__
        print(
            "mixed-only arguments: --ss_migration_contract PATH "
            "--slat_migration_contract PATH",
            file=sys.stderr,
        )
        _v2_infer.main()
        return

    ss_contract_path = _pop_argument("--ss_migration_contract")
    slat_contract_path = _pop_argument("--slat_migration_contract")
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
    ss_contract, slat_contract, _ = _validate_deployment(
        pretrained=pretrained,
        ss_path=ss_path,
        slat_path=slat_path,
        stock_freeze_path=stock_path,
        ss_contract_path=ss_contract_path,
        slat_contract_path=slat_contract_path,
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

    def run_slat_no_vggt(**kwargs: Any):
        reports = original_run_slat(**kwargs)
        for report in reports:
            report.update(
                {
                    "format": REPORT_FORMAT,
                    "method": "native_no_vggt_mixed",
                    "ss_input_context_contract": NO_VGGT_MODEL_CONTRACT,
                    "slat_input_context_contract": NO_VGGT_SLAT_CONTRACT,
                    "vggt_model_loaded": False,
                    "vggt_model_executed": False,
                    "ss_migration_contract": migration_summary(ss_contract),
                    "slat_migration_contract": migration_summary(slat_contract),
                }
            )
            result_path = Path(report["mesh"]).parent / "result.json"
            atomic_json(result_path, report)
        return reports

    _v2_infer._run_slat = run_slat_no_vggt
    _v2_infer.main()

    manifest_path = Path(str(output_value)).expanduser().resolve() / "inference_manifest.json"
    manifest = load_json(manifest_path)
    manifest.update(
        {
            "format": MANIFEST_FORMAT,
            "method": "native_no_vggt_mixed",
            "ss_input_context_contract": NO_VGGT_MODEL_CONTRACT,
            "slat_input_context_contract": NO_VGGT_SLAT_CONTRACT,
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
            "ss_migration_contract": migration_summary(ss_contract),
            "slat_migration_contract": migration_summary(slat_contract),
        }
    )
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
