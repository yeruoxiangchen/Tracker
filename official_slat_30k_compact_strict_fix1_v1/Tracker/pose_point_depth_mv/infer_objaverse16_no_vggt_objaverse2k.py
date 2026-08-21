#!/usr/bin/env python3
"""Run a dev-selected Objaverse2K no-VGGT SLat on frozen Objaverse16."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv import infer_omni_real_native_v2 as _v2_infer
from pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat import (
    AGGREGATE_REPORT_FORMAT,
    upstream_binding,
)
from pose_point_depth_mv.freeze_objaverse16_test import PROTOCOL_FORMAT
from pose_point_depth_mv.infer_objaverse16_no_vggt_mixed import (
    MANIFEST_FORMAT,
    _load_model_sample,
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
from pose_point_depth_mv.no_vggt_ss_evidence import load_no_vggt_ss_evidence
from pose_point_depth_mv.objaverse2k_slat_pipeline import SPLIT_FORMAT
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.prepare_objaverse16_no_vggt_model_inputs import (
    MANIFEST_FORMAT as MODEL_INPUT_MANIFEST_FORMAT,
    OBJECT_FORMAT as MODEL_INPUT_OBJECT_FORMAT,
)


REPORT_FORMAT = "pose_point_depth_mv.objaverse16_no_vggt_objaverse2k_inference.v1"
METHOD = "native_no_vggt_objaverse2k_objaverse16"


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


def validate_objaverse2k_lineage(
    *,
    model_manifest_path: Path,
    ss_path: Path,
    slat_path: Path,
    stock_freeze_path: Path,
    ss_report_path: Path,
    dev_report_path: Path,
    pretrained: str,
) -> dict[str, Any]:
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
        or protocol.get("source_mesh_disjoint") is not True
    ):
        raise RuntimeError("Objaverse16 selection/source-mesh protocol did not pass")
    selected_objects = {
        str(row["object_uid"]) for row in selection.get("samples", [])
    }
    if len(selected_objects) != 16:
        raise RuntimeError("Objaverse16 selection does not contain 16 unique objects")

    ss_payload, ss_binding_all = load_no_vggt_ss_evidence(ss_report_path)
    del ss_payload
    ss_binding = upstream_binding(ss_binding_all)
    if (
        Path(ss_binding["checkpoint"]).resolve() != ss_path
        or ss_binding["checkpoint_sha256"] != sha256_file(ss_path)
    ):
        raise RuntimeError("requested frozen SS differs from its deployment report")
    ss_checkpoint = torch.load(ss_path, map_location="cpu")
    validate_native_ss_no_vggt_checkpoint(
        ss_checkpoint, pretrained=pretrained, allow_v2_parent=False
    )
    if ss_checkpoint.get("format") != NATIVE_SS_NO_VGGT_VERSION:
        raise RuntimeError("Objaverse16 requires a trained no-VGGT SS checkpoint")
    del ss_checkpoint

    slat_sha = sha256_file(slat_path)
    slat_checkpoint = torch.load(slat_path, map_location="cpu")
    if slat_checkpoint.get("format") != NATIVE_SLAT_NO_VGGT_VERSION:
        raise RuntimeError("Objaverse16 requires a trained no-VGGT SLat checkpoint")
    if dict(slat_checkpoint.get("model_summary", {}).get("upstream_native_ss", {})) != ss_binding:
        raise RuntimeError("Objaverse2K SLat and frozen SS deployments differ")
    if slat_checkpoint.get("args", {}).get("stock_context_views", "all") != "all":
        raise RuntimeError("Objaverse2K final diagnostic requires all-view Stock context")
    data_identity = dict(slat_checkpoint.get("data_identity", {}))
    training_objects = set(str(value) for value in data_identity.get("object_uids", []))
    if len(training_objects) != 2135:
        raise RuntimeError("Objaverse2K checkpoint does not bind 2135 training objects")
    overlap = sorted(selected_objects.intersection(training_objects))
    if overlap:
        raise RuntimeError(f"Objaverse16/Objaverse2K training overlap: {overlap}")
    training_manifest_path = Path(
        str(data_identity.get("lifting_cache_manifest", ""))
    ).expanduser().resolve()
    if (
        not training_manifest_path.is_file()
        or sha256_file(training_manifest_path)
        != data_identity.get("lifting_cache_manifest_sha256")
    ):
        raise RuntimeError("Objaverse2K training lifting manifest changed")
    training_manifest = load_json(training_manifest_path)
    split = dict(training_manifest.get("objaverse2k_split", {}))
    if (
        split.get("format") != SPLIT_FORMAT
        or split.get("name") != "train"
        or int(training_manifest.get("object_count", -1)) != 2135
        or set(
            str(row.get("object_uid", row["uid"]))
            for row in training_manifest.get("samples", [])
        )
        != training_objects
    ):
        raise RuntimeError("Objaverse2K training split identity differs")
    validate_native_slat_no_vggt_checkpoint(
        slat_checkpoint,
        pretrained=pretrained,
        stock_slat_freeze=load_stock_slat_freeze(stock_freeze_path),
        upstream_native_ss=ss_binding,
        allow_v2_parent=False,
    )
    del slat_checkpoint

    dev_report = load_json(dev_report_path)
    if (
        dev_report.get("format") != AGGREGATE_REPORT_FORMAT
        or dev_report.get("passed") is not True
        or dev_report.get("formal") is not False
        or int(dev_report.get("object_count", -1)) != 64
        or dev_report.get("checkpoints", {}).get("objaverse2k", {}).get("sha256")
        != slat_sha
    ):
        raise RuntimeError("Objaverse2K checkpoint was not selected by matched dev64")
    return {
        "training_regime": "objaverse2k_train2135_dev64",
        "training_object_count": 2135,
        "training_lifting_manifest": str(training_manifest_path),
        "training_lifting_manifest_sha256": sha256_file(training_manifest_path),
        "training_object_overlap": [],
        "training_object_disjoint": True,
        "source_mesh_disjoint": True,
        "native_ss_report": str(ss_report_path),
        "native_ss_report_sha256": sha256_file(ss_report_path),
        "dev64_selection_report": str(dev_report_path),
        "dev64_selection_report_sha256": sha256_file(dev_report_path),
        "native_ss_checkpoint_sha256": sha256_file(ss_path),
        "native_slat_checkpoint_sha256": slat_sha,
    }


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Objaverse2K lineage arguments: --native_ss_report PATH "
            "--slat_dev_report PATH",
            file=sys.stderr,
        )
        _v2_infer.__doc__ = __doc__
        _v2_infer.main()
        return
    ss_report_path = Path(_pop_argument("--native_ss_report")).expanduser().resolve()
    dev_report_path = Path(_pop_argument("--slat_dev_report")).expanduser().resolve()
    model_value = _argument("--model_input_manifest")
    ss_value = _argument("--native_ss_checkpoint")
    slat_value = _argument("--native_slat_checkpoint")
    stock_value = _argument("--stock_slat_freeze")
    output_value = _argument("--output_dir")
    pretrained = _argument("--pretrained") or "Stable-X/trellis-vggt-v0-2"
    if not all((model_value, ss_value, slat_value, stock_value, output_value)):
        raise ValueError("model input, SS, SLat, stock freeze, and output_dir are required")
    model_path = Path(str(model_value)).expanduser().resolve()
    ss_path = Path(str(ss_value)).expanduser().resolve()
    slat_path = Path(str(slat_value)).expanduser().resolve()
    stock_path = Path(str(stock_value)).expanduser().resolve()
    output_dir = Path(str(output_value)).expanduser().resolve()
    lineage = validate_objaverse2k_lineage(
        model_manifest_path=model_path,
        ss_path=ss_path,
        slat_path=slat_path,
        stock_freeze_path=stock_path,
        ss_report_path=ss_report_path,
        dev_report_path=dev_report_path,
        pretrained=pretrained,
    )

    _v2_infer.MODEL_INPUT_MANIFEST_FORMAT = MODEL_INPUT_MANIFEST_FORMAT
    _v2_infer.MODEL_INPUT_OBJECT_FORMAT = MODEL_INPUT_OBJECT_FORMAT
    _v2_infer.REPORT_FORMAT = REPORT_FORMAT
    # Preserve the existing evaluator's accepted target-free manifest format.
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

    def run_slat_objaverse2k(**kwargs: Any):
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
                    "objaverse2k_lineage": lineage,
                }
            )
            atomic_json(Path(report["mesh"]).parent / "result.json", report)
        return reports

    _v2_infer._run_slat = run_slat_objaverse2k
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
            "objaverse2k_lineage": lineage,
        }
    )
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
