#!/usr/bin/env python3
"""Run the contracted mixed no-VGGT deployment on frozen Objaverse16 inputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv import infer_omni_real_native_v2 as _v2_infer
from pose_point_depth_mv.infer_omni_real_native_no_vggt_mixed import (
    _validate_deployment,
)
from pose_point_depth_mv.native_slat_genrecon_no_vggt import (
    NO_VGGT_SLAT_CONTRACT,
    build_native_slat_no_vggt_components,
    validate_native_slat_no_vggt_checkpoint,
)
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    NO_VGGT_MODEL_CONTRACT,
    build_native_ss_no_vggt_components,
    validate_native_ss_no_vggt_checkpoint,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    object_key,
    validate_bound_file,
)
from pose_point_depth_mv.prepare_objaverse16_no_vggt_model_inputs import (
    FORBIDDEN_MODEL_FIELDS,
    MANIFEST_FORMAT as MODEL_INPUT_MANIFEST_FORMAT,
    OBJECT_FORMAT as MODEL_INPUT_OBJECT_FORMAT,
)
from pose_point_depth_mv.real_full_no_vggt_migration import (
    migration_summary,
)


REPORT_FORMAT = "pose_point_depth_mv.objaverse16_no_vggt_mixed_inference.v1"
MANIFEST_FORMAT = (
    "pose_point_depth_mv.objaverse16_no_vggt_mixed_inference_manifest.v1"
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


def _load_model_sample(row: dict[str, Any]) -> dict[str, Any]:
    path = validate_bound_file(
        row["model_input"], row["model_input_sha256"], label="Objaverse model input"
    )
    payload = torch.load(path, map_location="cpu")
    if (
        payload.get("format") != MODEL_INPUT_OBJECT_FORMAT
        or payload.get("object_key") != object_key(row)
        or payload.get("condition_sha256") != row.get("condition_sha256")
    ):
        raise RuntimeError(f"Objaverse16 model input identity differs: {path}")
    leaked = sorted(FORBIDDEN_MODEL_FIELDS & set(payload))
    if leaked:
        raise RuntimeError(f"Objaverse16 inference input leaked forbidden fields: {leaked}")
    guards = {
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "point_cloud_tensor_present": False,
        "point_cloud_consumed": False,
        "target_or_mesh_consumed": False,
    }
    mismatch = {
        key: (payload.get(key), expected)
        for key, expected in guards.items()
        if payload.get(key) is not expected
    }
    if mismatch:
        raise RuntimeError(f"Objaverse16 inference scope guard differs: {mismatch}")
    views = int(payload["visual_patch_features"].shape[0])
    height, width = map(int, payload["projection_image_size"])
    payload["predicted_depth"] = torch.zeros((), dtype=torch.float16).expand(
        views, height, width
    )
    return payload


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Objaverse16-only arguments: --ss_migration_contract PATH "
            "--slat_migration_contract PATH",
            file=sys.stderr,
        )
        _v2_infer.__doc__ = __doc__
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
    _v2_infer._load_model_sample = _load_model_sample
    _v2_infer.validate_native_ss_genrecon_checkpoint = (
        validate_native_ss_no_vggt_checkpoint
    )
    _v2_infer.build_native_ss_genrecon_components = build_native_ss_no_vggt_components
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
                    "method": "native_no_vggt_mixed_objaverse16",
                    "protocol_scope": "frozen_objaverse_test16",
                    "formal": False,
                    "ss_input_context_contract": NO_VGGT_MODEL_CONTRACT,
                    "slat_input_context_contract": NO_VGGT_SLAT_CONTRACT,
                    "vggt_model_loaded": False,
                    "vggt_model_executed": False,
                    "point_cloud_tensor_present": False,
                    "point_cloud_consumed": False,
                    "target_or_metric_consumed": False,
                    "ss_migration_contract": migration_summary(ss_contract),
                    "slat_migration_contract": migration_summary(slat_contract),
                }
            )
            result_path = Path(report["mesh"]).parent / "result.json"
            atomic_json(result_path, report)
        return reports

    _v2_infer._run_slat = run_slat_no_vggt
    _v2_infer.__doc__ = __doc__
    _v2_infer.main()

    manifest_path = Path(str(output_value)).expanduser().resolve() / "inference_manifest.json"
    manifest = load_json(manifest_path)
    manifest.update(
        {
            "format": MANIFEST_FORMAT,
            "method": "native_no_vggt_mixed_objaverse16",
            "protocol_scope": "frozen_objaverse_test16",
            "formal": False,
            "ss_input_context_contract": NO_VGGT_MODEL_CONTRACT,
            "slat_input_context_contract": NO_VGGT_SLAT_CONTRACT,
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
            "point_cloud_tensor_present": False,
            "point_cloud_consumed": False,
            "target_or_metric_consumed": False,
            "ss_migration_contract": migration_summary(ss_contract),
            "slat_migration_contract": migration_summary(slat_contract),
        }
    )
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
