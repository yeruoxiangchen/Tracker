#!/usr/bin/env python3
"""Run no-VGGT Native SS + Native-SLat on DINO-only runtime-O inputs."""

from __future__ import annotations

import sys
from pathlib import Path

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
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    NATIVE_SS_NO_VGGT_VERSION,
    NO_VGGT_MODEL_CONTRACT,
    build_native_ss_no_vggt_components,
    validate_native_ss_no_vggt_checkpoint,
)
from pose_point_depth_mv.omni_real_benchmark_common import atomic_json, load_json


REPORT_FORMAT = "pose_point_depth_mv.omni_real_native_no_vggt_inference.v1"
MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_native_no_vggt_inference_manifest.v1"
)


def _argument(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        prefix = f"{name}="
        return next((value[len(prefix) :] for value in sys.argv if value.startswith(prefix)), None)


def main() -> None:
    help_requested = "--help" in sys.argv or "-h" in sys.argv
    ss_path = _argument("--native_ss_checkpoint")
    slat_path = _argument("--native_slat_checkpoint")
    output_value = _argument("--output_dir")
    pretrained = _argument("--pretrained") or "Stable-X/trellis-vggt-v0-2"
    if not help_requested and (not ss_path or not slat_path or not output_value):
        raise ValueError("SS checkpoint, SLat checkpoint, and output_dir are required")
    if not help_requested:
        ss_header = torch.load(ss_path, map_location="cpu")
        validate_native_ss_no_vggt_checkpoint(
            ss_header, pretrained=pretrained, allow_v2_parent=False
        )
        if ss_header.get("format") != NATIVE_SS_NO_VGGT_VERSION:
            raise ValueError("inference requires a trained no-VGGT Native SS checkpoint")
        slat_header = torch.load(slat_path, map_location="cpu")
        if slat_header.get("format") != NATIVE_SLAT_NO_VGGT_VERSION:
            raise ValueError("inference requires a trained no-VGGT Native-SLat checkpoint")
        del ss_header, slat_header

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

    original_run_ss = _v2_infer._run_ss
    original_run_slat = _v2_infer._run_slat

    def run_ss_no_vggt(**kwargs):
        result = original_run_ss(**kwargs)
        for row in kwargs["rows"]:
            for seed in kwargs["seeds"]:
                _, report_path = _v2_infer._coord_paths(
                    kwargs["output_dir"], row, seed
                )
                report = load_json(report_path)
                report["input_context_contract"] = NO_VGGT_MODEL_CONTRACT
                report["vggt_model_executed"] = False
                atomic_json(report_path, report)
        return result

    def run_slat_no_vggt(**kwargs):
        reports = original_run_slat(**kwargs)
        for report in reports:
            report["method"] = "native_no_vggt"
            report["ss_input_context_contract"] = NO_VGGT_MODEL_CONTRACT
            report["slat_input_context_contract"] = NO_VGGT_SLAT_CONTRACT
            report["vggt_model_executed"] = False
            result_path = Path(report["mesh"]).parent / "result.json"
            atomic_json(result_path, report)
        return reports

    _v2_infer._run_ss = run_ss_no_vggt
    _v2_infer._run_slat = run_slat_no_vggt
    _v2_infer.main()

    if help_requested:
        return
    manifest_path = Path(output_value).expanduser().resolve() / "inference_manifest.json"
    manifest = load_json(manifest_path)
    manifest.update(
        {
            "format": MANIFEST_FORMAT,
            "method": "native_no_vggt",
            "ss_input_context_contract": NO_VGGT_MODEL_CONTRACT,
            "slat_input_context_contract": NO_VGGT_SLAT_CONTRACT,
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
        }
    )
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
