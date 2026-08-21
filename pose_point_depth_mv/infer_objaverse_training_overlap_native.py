#!/usr/bin/env python3
"""Run Objaverse2K-SLat or M8 on an Objaverse training-overlap subset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv import infer_omni_real_native_v2 as _v2_infer
from pose_point_depth_mv.infer_objaverse16_no_vggt_mixed import _load_model_sample
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
from pose_point_depth_mv.prepare_objaverse16_no_vggt_model_inputs import (
    MANIFEST_FORMAT as MODEL_INPUT_MANIFEST_FORMAT,
    OBJECT_FORMAT as MODEL_INPUT_OBJECT_FORMAT,
)
from pose_point_depth_mv.training_overlap_objaverse import (
    TRAINING_OVERLAP_SCOPE,
    validate_selection,
)


REPORT_FORMAT = "pose_point_depth_mv.objaverse_training_overlap_native_inference.v1"
MANIFEST_FORMAT = (
    "pose_point_depth_mv.objaverse_training_overlap_native_inference_manifest.v1"
)
MODEL_LABELS = ("objaverse2k_slat", "m8")


def _argument(name: str) -> str | None:
    prefix = f"{name}="
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return next((value[len(prefix) :] for value in sys.argv if value.startswith(prefix)), None)


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


def _source_slat_is_bound(
    *, model_label: str, checkpoint_identity: dict[str, Any], selection: dict[str, Any]
) -> bool:
    protocol = selection["training_overlap_protocol"]
    source_path = Path(str(protocol["source_slat_manifest"])).resolve()
    source_sha = str(protocol["source_slat_manifest_sha256"])
    checkpoint_path = Path(str(checkpoint_identity.get("cache_manifest", ""))).resolve()
    checkpoint_sha = str(checkpoint_identity.get("cache_manifest_sha256", ""))
    if model_label == "objaverse2k_slat":
        return checkpoint_path == source_path and checkpoint_sha == source_sha
    parent = load_json(checkpoint_path)
    if sha256_file(checkpoint_path) != checkpoint_sha:
        return False
    synthetic = [
        row for row in parent.get("domains", []) if str(row.get("name")) == "synthetic"
    ]
    return (
        len(synthetic) == 1
        and Path(str(synthetic[0].get("manifest", ""))).resolve() == source_path
        and str(synthetic[0].get("manifest_sha256", "")) == source_sha
    )


def validate_lineage(
    *,
    model_label: str,
    model_manifest_path: Path,
    ss_path: Path,
    slat_path: Path,
    stock_path: Path,
    pretrained: str,
) -> dict[str, Any]:
    if model_label not in MODEL_LABELS:
        raise ValueError(f"unsupported model_label={model_label!r}")
    model_manifest = load_json(model_manifest_path)
    if (
        model_manifest.get("format") != MODEL_INPUT_MANIFEST_FORMAT
        or model_manifest.get("passed") is not True
        or model_manifest.get("protocol_scope") != TRAINING_OVERLAP_SCOPE
        or model_manifest.get("training_overlap") is not True
    ):
        raise RuntimeError("model inputs are not a passed training-overlap subset")
    selection_path = Path(str(model_manifest["selection_manifest"])).resolve()
    if sha256_file(selection_path) != model_manifest.get("selection_manifest_sha256"):
        raise RuntimeError("model-input selection binding changed")
    selection = load_json(selection_path)
    contract = validate_selection(selection)
    expected_source = (
        "objaverse2k_train" if model_label == "objaverse2k_slat" else "mixed_objaverse_train"
    )
    if contract.source_scope != expected_source:
        raise RuntimeError(f"{model_label} cannot consume source_scope={contract.source_scope}")

    ss_checkpoint = torch.load(ss_path, map_location="cpu", weights_only=False)
    slat_checkpoint = torch.load(slat_path, map_location="cpu", weights_only=False)
    if ss_checkpoint.get("format") != NATIVE_SS_NO_VGGT_VERSION:
        raise RuntimeError("Native SS checkpoint is not trained no-VGGT format")
    if slat_checkpoint.get("format") != NATIVE_SLAT_NO_VGGT_VERSION:
        raise RuntimeError("Native SLat checkpoint is not trained no-VGGT format")
    validate_native_ss_no_vggt_checkpoint(
        ss_checkpoint, pretrained=pretrained, allow_v2_parent=False
    )
    upstream = dict(slat_checkpoint.get("model_summary", {}).get("upstream_native_ss", {}))
    if upstream.get("checkpoint_sha256") != sha256_file(ss_path):
        raise RuntimeError("SLat upstream Native SS differs from requested checkpoint")
    stock = load_stock_slat_freeze(stock_path)
    validate_native_slat_no_vggt_checkpoint(
        slat_checkpoint,
        pretrained=pretrained,
        stock_slat_freeze=stock,
        upstream_native_ss=upstream,
        allow_v2_parent=False,
    )
    identity = dict(slat_checkpoint.get("data_identity", {}))
    selected_objects = set(contract.selected_object_uids)
    training_objects = {str(value) for value in identity.get("object_uids", [])}
    if not selected_objects or not selected_objects.issubset(training_objects):
        raise RuntimeError("selected objects are not all bound SLat training objects")
    if not _source_slat_is_bound(
        model_label=model_label, checkpoint_identity=identity, selection=selection
    ):
        raise RuntimeError("selected source SLat cache is not bound to the checkpoint")
    stock_summary = dict(slat_checkpoint.get("model_summary", {}).get("stock_slat_freeze", {}))
    if (
        Path(str(stock_summary.get("path", ""))).resolve() != stock_path
        or stock_summary.get("file_sha256") != sha256_file(stock_path)
        or identity.get("stock_slat_freeze_sha256") != stock_summary.get("freeze_sha256")
    ):
        raise RuntimeError("checkpoint Stock SLat freeze binding differs")
    return {
        "model_label": model_label,
        "source_scope": contract.source_scope,
        "training_overlap": True,
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "training_object_count": len(training_objects),
        "selected_training_object_count": len(selected_objects),
        "selected_objects_are_training_objects": True,
        "native_ss_checkpoint_sha256": sha256_file(ss_path),
        "native_slat_checkpoint_sha256": sha256_file(slat_path),
        "stock_slat_freeze_sha256": sha256_file(stock_path),
        "stock_slat_freeze_contract_sha256": stock_summary["freeze_sha256"],
        "passed": True,
    }


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "training-overlap arguments: --model_label {objaverse2k_slat,m8} "
            "--worker_index N --num_workers N",
            file=sys.stderr,
        )
        _v2_infer.__doc__ = __doc__
        _v2_infer.main()
        return
    model_label = _pop_argument("--model_label")
    worker_index = int(_pop_argument("--worker_index"))
    num_workers = int(_pop_argument("--num_workers"))
    if num_workers <= 0 or not 0 <= worker_index < num_workers:
        raise ValueError("worker_index must be in [0, num_workers)")
    model_value = _argument("--model_input_manifest")
    ss_value = _argument("--native_ss_checkpoint")
    slat_value = _argument("--native_slat_checkpoint")
    stock_value = _argument("--stock_slat_freeze")
    output_value = _argument("--output_dir")
    pretrained = _argument("--pretrained") or "Stable-X/trellis-vggt-v0-2"
    if not all((model_value, ss_value, slat_value, stock_value, output_value)):
        raise ValueError("model input, SS, SLat, Stock freeze, and output are required")
    model_path = Path(str(model_value)).expanduser().resolve()
    ss_path = Path(str(ss_value)).expanduser().resolve()
    slat_path = Path(str(slat_value)).expanduser().resolve()
    stock_path = Path(str(stock_value)).expanduser().resolve()
    output_dir = Path(str(output_value)).expanduser().resolve()
    lineage = validate_lineage(
        model_label=model_label,
        model_manifest_path=model_path,
        ss_path=ss_path,
        slat_path=slat_path,
        stock_path=stock_path,
        pretrained=pretrained,
    )
    model_manifest = load_json(model_path)
    rows = sorted(model_manifest["objects"], key=lambda row: str(row["object_id"]))
    selected_rows = [
        row for position, row in enumerate(rows) if position % num_workers == worker_index
    ]
    if not selected_rows:
        raise RuntimeError("Native worker shard contains no objects")
    for row in selected_rows:
        sys.argv.extend(("--object", str(row["object_key"])))
    global_positions = {
        str(row["object_key"]): position for position, row in enumerate(rows)
    }

    def noise_object_position(row: dict[str, Any], fallback: int) -> int:
        del fallback
        return int(global_positions[str(row["object_key"])])

    _v2_infer.MODEL_INPUT_MANIFEST_FORMAT = MODEL_INPUT_MANIFEST_FORMAT
    _v2_infer.MODEL_INPUT_OBJECT_FORMAT = MODEL_INPUT_OBJECT_FORMAT
    _v2_infer.REPORT_FORMAT = REPORT_FORMAT
    _v2_infer.MANIFEST_FORMAT = MANIFEST_FORMAT
    _v2_infer._load_model_sample = _load_model_sample
    _v2_infer._noise_object_position = noise_object_position
    _v2_infer.validate_native_ss_genrecon_checkpoint = validate_native_ss_no_vggt_checkpoint
    _v2_infer.build_native_ss_genrecon_components = build_native_ss_no_vggt_components
    _v2_infer.validate_native_slat_genrecon_v2_checkpoint = (
        validate_native_slat_no_vggt_checkpoint
    )
    _v2_infer.build_native_slat_genrecon_v2_components = build_native_slat_no_vggt_components
    original_run_slat = _v2_infer._run_slat

    def run_slat_overlap(**kwargs: Any):
        reports = original_run_slat(**kwargs)
        for report in reports:
            report.update(
                {
                    "format": REPORT_FORMAT,
                    "method": model_label,
                    "protocol_scope": TRAINING_OVERLAP_SCOPE,
                    "formal": False,
                    "training_overlap": True,
                    "training_object_disjoint": False,
                    "source_mesh_disjoint": False,
                    "source_scope": lineage["source_scope"],
                    "ss_input_context_contract": NO_VGGT_MODEL_CONTRACT,
                    "slat_input_context_contract": NO_VGGT_SLAT_CONTRACT,
                    "vggt_model_loaded": False,
                    "vggt_model_executed": False,
                    "point_cloud_tensor_present": False,
                    "point_cloud_consumed": False,
                    "target_or_metric_consumed": False,
                    "training_lineage": lineage,
                    "noise_object_position": global_positions[str(report["object_key"])],
                    "noise_identity_independent_of_worker_count": True,
                    "output_frame": "latent decoder canonical; transform_pose=False",
                }
            )
            atomic_json(Path(report["mesh"]).parent / "result.json", report)
        return reports

    _v2_infer._run_slat = run_slat_overlap
    _v2_infer.__doc__ = __doc__
    _v2_infer.main()
    manifest_path = output_dir / "inference_manifest.json"
    manifest = load_json(manifest_path)
    manifest.update(
        {
            "format": MANIFEST_FORMAT,
            "method": model_label,
            "protocol_scope": TRAINING_OVERLAP_SCOPE,
            "formal": False,
            "training_overlap": True,
            "training_object_disjoint": False,
            "source_mesh_disjoint": False,
            "source_scope": lineage["source_scope"],
            "selection_object_count": len(rows),
            "worker_index": worker_index,
            "num_workers": num_workers,
            "ss_input_context_contract": NO_VGGT_MODEL_CONTRACT,
            "slat_input_context_contract": NO_VGGT_SLAT_CONTRACT,
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
            "point_cloud_tensor_present": False,
            "point_cloud_consumed": False,
            "target_or_metric_consumed": False,
            "training_lineage": lineage,
            "noise_identity_independent_of_worker_count": True,
            "output_frame": "latent decoder canonical; transform_pose=False",
        }
    )
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
