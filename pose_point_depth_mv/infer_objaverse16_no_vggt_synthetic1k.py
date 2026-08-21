#!/usr/bin/env python3
"""Run the frozen synthetic1k No-VGGT SS/SLat deployment on Objaverse16."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv import infer_omni_real_native_v2 as _v2_infer
from pose_point_depth_mv.freeze_objaverse16_test import (
    PROTOCOL_FORMAT,
    sha256_file,
)
from pose_point_depth_mv.infer_objaverse16_no_vggt_mixed import _load_model_sample
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
from pose_point_depth_mv.prepare_objaverse16_no_vggt_model_inputs import (
    MANIFEST_FORMAT as MODEL_INPUT_MANIFEST_FORMAT,
    OBJECT_FORMAT as MODEL_INPUT_OBJECT_FORMAT,
)


REPORT_FORMAT = "pose_point_depth_mv.objaverse16_no_vggt_synthetic1k_inference.v1"
MANIFEST_FORMAT = (
    "pose_point_depth_mv.objaverse16_no_vggt_synthetic1k_inference_manifest.v1"
)
METHOD = "native_no_vggt_synthetic1k_objaverse16"
SS_REPORT_FORMAT = "pose_point_depth_mv.native_ss_no_vggt_eval.v1"
SLAT_REPORT_FORMAT = "pose_point_depth_mv.native_slat_genrecon_no_vggt.v1"


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


def _same_path(left: str, right: Path) -> bool:
    return Path(left).expanduser().resolve() == right


def validate_synthetic1k_lineage(
    *,
    model_manifest_path: Path,
    ss_path: Path,
    slat_path: Path,
    ss_report_path: Path,
    slat_report_path: Path,
    pretrained: str,
) -> dict[str, Any]:
    """Validate checkpoint/report lineage and the frozen training-disjoint audit."""

    ss_sha = sha256_file(ss_path)
    slat_sha = sha256_file(slat_path)
    ss_report_sha = sha256_file(ss_report_path)
    slat_report_sha = sha256_file(slat_report_path)
    ss_report = load_json(ss_report_path)
    slat_report = load_json(slat_report_path)
    ss_protocol = dict(ss_report.get("protocol", {}))
    if (
        ss_report.get("format") != SS_REPORT_FORMAT
        or ss_report.get("passed") is not True
        or int(ss_protocol.get("object_start", -1)) != 32
        or int(ss_protocol.get("object_end", -1)) != 64
        or ss_protocol.get("checkpoint_sha256") != ss_sha
    ):
        raise RuntimeError("synthetic1k SS final32 report/checkpoint binding differs")

    slat_summary = dict(slat_report.get("model_summary", {}))
    upstream = dict(slat_summary.get("upstream_native_ss", {}))
    if (
        slat_report.get("format") != SLAT_REPORT_FORMAT
        or slat_report.get("completed") is not True
        or slat_report.get("passed") is not True
        or int(slat_report.get("step", -1)) != 2000
        or upstream.get("checkpoint_sha256") != ss_sha
        or upstream.get("report_sha256") != ss_report_sha
        or upstream.get("weights") != "ema"
        or float(upstream.get("cfg_strength", -1.0)) != 5.0
    ):
        raise RuntimeError("synthetic1k SLat report/upstream SS binding differs")

    ss_header = torch.load(ss_path, map_location="cpu")
    validate_native_ss_no_vggt_checkpoint(
        ss_header, pretrained=pretrained, allow_v2_parent=False
    )
    if (
        ss_header.get("format") != NATIVE_SS_NO_VGGT_VERSION
        or int(ss_header.get("step", -1)) != 2000
    ):
        raise RuntimeError("synthetic1k SS checkpoint is not the frozen step2000 model")
    del ss_header
    slat_header = torch.load(slat_path, map_location="cpu")
    slat_header_summary = dict(slat_header.get("model_summary", {}))
    if (
        slat_header.get("format") != NATIVE_SLAT_NO_VGGT_VERSION
        or int(slat_header.get("step", -1)) != 2000
        or dict(slat_header_summary.get("upstream_native_ss", {})).get(
            "checkpoint_sha256"
        )
        != ss_sha
    ):
        raise RuntimeError("synthetic1k SLat checkpoint lineage differs")
    del slat_header

    model_manifest = load_json(model_manifest_path)
    if (
        model_manifest.get("format") != MODEL_INPUT_MANIFEST_FORMAT
        or model_manifest.get("protocol_scope") != "frozen_objaverse_test16"
        or model_manifest.get("passed") is not True
        or int(model_manifest.get("selected_object_count", -1)) != 16
    ):
        raise RuntimeError("Objaverse16 target-free model-input manifest differs")
    selection_path = Path(model_manifest["selection_manifest"]).expanduser().resolve()
    if model_manifest.get("selection_manifest_sha256") != sha256_file(selection_path):
        raise RuntimeError("Objaverse16 selection binding differs")
    selection = load_json(selection_path)
    protocol = dict(selection.get("objaverse16_protocol", {}))
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("passed") is not True
        or protocol.get("training_object_disjoint") is not True
        or protocol.get("source_mesh_disjoint") is not True
    ):
        raise RuntimeError("Objaverse16 selection/disjoint protocol did not pass")

    data_identity = dict(slat_summary.get("data_identity", {}))
    training_manifest = Path(
        str(data_identity.get("lifting_cache_manifest", ""))
    ).expanduser().resolve()
    training_sha = str(data_identity.get("lifting_cache_manifest_sha256", ""))
    matching_audits = [
        dict(audit)
        for audit in protocol.get("deployment_training_audits", [])
        if _same_path(str(audit.get("manifest", "")), training_manifest)
        and audit.get("manifest_sha256") == training_sha
    ]
    if (
        len(matching_audits) != 1
        or matching_audits[0].get("passed") is not True
        or matching_audits[0].get("selected_object_overlap") != []
        or int(matching_audits[0].get("object_count", -1)) != 868
    ):
        raise RuntimeError(
            "Objaverse16 selection lacks the exact synthetic1k train868 disjoint audit"
        )

    return {
        "training_regime": "reviewed1k_synthetic_train868",
        "training_object_count": 868,
        "training_lifting_manifest": str(training_manifest),
        "training_lifting_manifest_sha256": training_sha,
        "training_object_overlap": [],
        "training_object_disjoint": True,
        "source_mesh_disjoint": True,
        "ss_report": str(ss_report_path),
        "ss_report_sha256": ss_report_sha,
        "slat_report": str(slat_report_path),
        "slat_report_sha256": slat_report_sha,
        "native_ss_checkpoint_sha256": ss_sha,
        "native_slat_checkpoint_sha256": slat_sha,
    }


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Objaverse16 synthetic1k arguments: --native_ss_report PATH "
            "--native_slat_report PATH",
            file=sys.stderr,
        )
        _v2_infer.__doc__ = __doc__
        _v2_infer.main()
        return

    ss_report_path = Path(_pop_argument("--native_ss_report")).expanduser().resolve()
    slat_report_path = Path(_pop_argument("--native_slat_report")).expanduser().resolve()
    model_value = _argument("--model_input_manifest")
    ss_value = _argument("--native_ss_checkpoint")
    slat_value = _argument("--native_slat_checkpoint")
    output_value = _argument("--output_dir")
    pretrained = _argument("--pretrained") or "Stable-X/trellis-vggt-v0-2"
    if not all((model_value, ss_value, slat_value, output_value)):
        raise ValueError("model input, SS, SLat, and output_dir are required")
    model_path = Path(str(model_value)).expanduser().resolve()
    ss_path = Path(str(ss_value)).expanduser().resolve()
    slat_path = Path(str(slat_value)).expanduser().resolve()
    output_dir = Path(str(output_value)).expanduser().resolve()
    lineage = validate_synthetic1k_lineage(
        model_manifest_path=model_path,
        ss_path=ss_path,
        slat_path=slat_path,
        ss_report_path=ss_report_path,
        slat_report_path=slat_report_path,
        pretrained=pretrained,
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

    def run_slat_synthetic1k(**kwargs: Any):
        reports = original_run_slat(**kwargs)
        for report in reports:
            report.update(
                {
                    "format": REPORT_FORMAT,
                    "method": METHOD,
                    "protocol_scope": "frozen_objaverse_test16",
                    "formal": False,
                    "training_regime": lineage["training_regime"],
                    "training_object_disjoint": True,
                    "source_mesh_disjoint": True,
                    "ss_input_context_contract": NO_VGGT_MODEL_CONTRACT,
                    "slat_input_context_contract": NO_VGGT_SLAT_CONTRACT,
                    "vggt_model_loaded": False,
                    "vggt_model_executed": False,
                    "point_cloud_tensor_present": False,
                    "point_cloud_consumed": False,
                    "target_or_metric_consumed": False,
                    "synthetic1k_lineage": lineage,
                }
            )
            result_path = Path(report["mesh"]).parent / "result.json"
            atomic_json(result_path, report)
        return reports

    _v2_infer._run_slat = run_slat_synthetic1k
    _v2_infer.__doc__ = __doc__
    _v2_infer.main()

    manifest_path = output_dir / "inference_manifest.json"
    manifest = load_json(manifest_path)
    manifest.update(
        {
            "format": MANIFEST_FORMAT,
            "method": METHOD,
            "protocol_scope": "frozen_objaverse_test16",
            "formal": False,
            "training_regime": lineage["training_regime"],
            "training_object_disjoint": True,
            "source_mesh_disjoint": True,
            "ss_input_context_contract": NO_VGGT_MODEL_CONTRACT,
            "slat_input_context_contract": NO_VGGT_SLAT_CONTRACT,
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
            "point_cloud_tensor_present": False,
            "point_cloud_consumed": False,
            "target_or_metric_consumed": False,
            "synthetic1k_lineage": lineage,
        }
    )
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
