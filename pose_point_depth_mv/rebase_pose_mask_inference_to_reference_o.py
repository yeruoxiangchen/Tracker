#!/usr/bin/env python3
"""Rebase pose+mask predictions through W into a frozen reference runtime-O frame."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import utc_now
from pose_point_depth_mv.evaluate_omni_real_mesh_benchmark import load_mesh
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    canonical_sha256,
    index_objects,
    load_json,
    object_key,
    sha256_file,
    validate_bound_file,
)
from pose_point_depth_mv.real_object_canonicalization import (
    similarity_scale,
    validate_proper_similarity,
)


REPORT_FORMAT = "pose_point_depth_mv.pose_mask_rebased_inference.v1"
MANIFEST_FORMAT = "pose_point_depth_mv.pose_mask_rebased_inference_manifest.v1"
POSE_MASK_RUNTIME_VARIANT = "pose_point_depth_mv.omni_real_pose_mask_runtime_ablation.v1"
NO_VGGT_INFERENCE_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_native_no_vggt_mixed_inference_manifest.v1"
)
PROTOCOL_SCOPES = ("development", "formal_holdout64_blind_addendum")


def object_frame_rebase_transform(
    pose_mask_T_O2W: np.ndarray, reference_T_W2O: np.ndarray
) -> np.ndarray:
    """Return the observable similarity mapping O_posemask directly to O_reference."""

    alt = validate_proper_similarity(pose_mask_T_O2W, name="pose_mask_T_O2W")
    ref = validate_proper_similarity(reference_T_W2O, name="reference_T_W2O")
    return validate_proper_similarity(ref @ alt, name="T_pose_mask_O_to_reference_O")


def _load_frame(cache_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as payload:
        missing = {"T_O2W", "T_W2O"}.difference(payload.files)
        if missing:
            raise RuntimeError(f"runtime cache lacks frame fields {sorted(missing)}")
        T_O2W = np.asarray(payload["T_O2W"], dtype=np.float64)
        T_W2O = np.asarray(payload["T_W2O"], dtype=np.float64)
    validate_proper_similarity(T_O2W, name="T_O2W")
    validate_proper_similarity(T_W2O, name="T_W2O")
    roundtrip = float(np.max(np.abs(T_W2O @ T_O2W - np.eye(4))))
    if roundtrip > 1.0e-8:
        raise RuntimeError(f"runtime frame inverse roundtrip failed: {roundtrip}")
    return T_O2W, T_W2O


def _atomic_export_obj(mesh: trimesh.Trimesh, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{os.getpid()}{destination.suffix}"
    )
    mesh.export(temporary, file_type="obj")
    os.replace(temporary, destination)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_mask_inference_manifest", required=True)
    parser.add_argument("--pose_mask_runtime_manifest", required=True)
    parser.add_argument("--reference_runtime_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--protocol_scope", choices=PROTOCOL_SCOPES, default="development"
    )
    parser.add_argument("--expected_objects", type=int, default=0)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    inference_path = Path(args.pose_mask_inference_manifest).expanduser().resolve()
    pose_runtime_path = Path(args.pose_mask_runtime_manifest).expanduser().resolve()
    reference_runtime_path = Path(args.reference_runtime_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    inference = load_json(inference_path)
    pose_runtime = load_json(pose_runtime_path)
    reference_runtime = load_json(reference_runtime_path)
    formal = args.protocol_scope == "formal_holdout64_blind_addendum"
    expected_objects = int(args.expected_objects)
    if formal and expected_objects != 64:
        raise ValueError("formal Holdout64 rebase requires --expected_objects 64")
    if not formal and expected_objects < 0:
        raise ValueError("expected_objects cannot be negative")
    if (
        inference.get("format") != NO_VGGT_INFERENCE_MANIFEST_FORMAT
        or inference.get("passed") is not True
        or inference.get("method") != "native_no_vggt_mixed"
        or inference.get("target_or_metric_consumed") is not False
    ):
        raise RuntimeError(f"pose+mask inference manifest did not pass: {inference_path}")
    if (
        pose_runtime.get("format") != RUNTIME_MANIFEST_FORMAT
        or pose_runtime.get("manifest_variant") != POSE_MASK_RUNTIME_VARIANT
        or pose_runtime.get("passed") is not True
        or pose_runtime.get("point_cloud_consumed") is not False
    ):
        raise RuntimeError(f"pose+mask runtime manifest did not pass: {pose_runtime_path}")
    if formal and (
        pose_runtime.get("formal") is not True
        or pose_runtime.get("protocol_scope")
        != "formal_holdout64_blind_addendum"
        or pose_runtime.get("gt_consumed") is not False
        or pose_runtime.get("old_mesh_consumed") is not False
        or pose_runtime.get("metric_or_ranking_consumed") is not False
        or pose_runtime.get("formal_holdout_binding", {}).get("passed") is not True
    ):
        raise RuntimeError("pose+mask runtime lacks the formal blind-addendum contract")
    if (
        reference_runtime.get("format") != RUNTIME_MANIFEST_FORMAT
        or reference_runtime.get("passed") is not True
    ):
        raise RuntimeError(f"reference runtime manifest did not pass: {reference_runtime_path}")
    if (
        Path(str(inference["runtime_input_manifest"])).resolve() != pose_runtime_path
        or str(inference["runtime_input_manifest_sha256"]) != sha256_file(pose_runtime_path)
    ):
        raise RuntimeError("pose+mask inference does not bind the requested runtime manifest")
    model_path = validate_bound_file(
        inference["model_input_manifest"],
        inference["model_input_manifest_sha256"],
        label="pose+mask DINO-only model input manifest",
    )
    model = load_json(model_path)
    if (
        Path(str(model["runtime_input_manifest"])).resolve() != pose_runtime_path
        or str(model["runtime_input_manifest_sha256"]) != sha256_file(pose_runtime_path)
        or model.get("vggt_model_loaded") is not False
        or model.get("vggt_model_executed") is not False
    ):
        raise RuntimeError("pose+mask DINO-only model input binding failed")

    pose_rows = index_objects(pose_runtime.get("objects", []), label="pose+mask runtime")
    reference_rows = index_objects(
        reference_runtime.get("objects", []), label="reference runtime"
    )
    inference_keys = {str(row["object_key"]) for row in inference.get("objects", [])}
    if inference_keys != set(pose_rows) or not inference_keys.issubset(reference_rows):
        raise RuntimeError("inference, pose+mask runtime, and reference object sets differ")
    if int(inference.get("object_count", -1)) != len(inference_keys):
        raise RuntimeError("pose+mask inference object count differs")
    if expected_objects and len(inference_keys) != expected_objects:
        raise RuntimeError("pose+mask inference count differs from protocol")
    inference_order = [str(row["object_key"]) for row in inference.get("objects", [])]
    pose_order = [object_key(row) for row in pose_runtime.get("objects", [])]
    reference_order = [object_key(row) for row in reference_runtime.get("objects", [])]
    if formal and (inference_order != pose_order or pose_order != reference_order):
        raise RuntimeError("formal Pose+Mask inference/reference object order differs")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for position, source in enumerate(inference.get("objects", []), start=1):
        key = str(source["object_key"])
        seed = int(source["seed"])
        pose_row = pose_rows[key]
        ref_row = reference_rows[key]
        pose_cache = validate_bound_file(
            pose_row["cache_npz"],
            pose_row["cache_npz_sha256"],
            label=f"pose+mask runtime cache {key}",
        )
        ref_cache = validate_bound_file(
            ref_row["cache_npz"],
            sha256_file(ref_row["cache_npz"]),
            label=f"reference runtime cache {key}",
        )
        pose_T_O2W, _ = _load_frame(pose_cache)
        _, reference_T_W2O = _load_frame(ref_cache)
        transform = object_frame_rebase_transform(pose_T_O2W, reference_T_W2O)
        inverse = np.linalg.inv(transform)
        roundtrip = float(np.max(np.abs(inverse @ transform - np.eye(4))))

        source_mesh = validate_bound_file(
            source["mesh"], source["mesh_sha256"], label=f"pose+mask mesh {key}/{seed}"
        )
        mesh = load_mesh(source_mesh)
        mesh.apply_transform(transform)
        destination = (
            output_dir
            / "meshes"
            / pose_row["category"]
            / pose_row["object_id"]
            / f"seed_{seed}"
            / "mesh_reference_o.obj"
        )
        _atomic_export_obj(mesh, destination)
        report = {
            "format": REPORT_FORMAT,
            "created_at_utc": utc_now(),
            "method": "native_no_vggt_pose_mask_rebased",
            "object_key": key,
            "category": str(pose_row["category"]),
            "object_id": str(pose_row["object_id"]),
            "seed": seed,
            "mesh": str(destination),
            "mesh_sha256": sha256_file(destination),
            "source_mesh": str(source_mesh),
            "source_mesh_sha256": str(source["mesh_sha256"]),
            "source_inference_record_sha256": canonical_sha256(source),
            "model_input": source["model_input"],
            "model_input_sha256": source["model_input_sha256"],
            "pose_mask_runtime_cache": str(pose_cache),
            "pose_mask_runtime_cache_sha256": sha256_file(pose_cache),
            "reference_runtime_cache": str(ref_cache),
            "reference_runtime_cache_sha256": sha256_file(ref_cache),
            "T_pose_mask_O_to_reference_O": transform.tolist(),
            "transform_scale": similarity_scale(transform),
            "transform_inverse_roundtrip_max_abs": roundtrip,
            "coordinate_chain": "O_posemask -> W -> O_reference",
            "gt_fit_or_icp_applied": False,
            "output_frame": "reference-runtime-O",
            "native_ss_checkpoint_sha256": source["native_ss_checkpoint_sha256"],
            "native_ss_weights": source["native_ss_weights"],
            "native_slat_checkpoint_sha256": source["native_slat_checkpoint_sha256"],
            "native_slat_weights": source["native_slat_weights"],
            "stock_slat_freeze_sha256": source["stock_slat_freeze_sha256"],
            "sampling": source["sampling"],
            "sampling_sha256": source["sampling_sha256"],
            "post_cfg_cap": source["post_cfg_cap"],
            "condition_scale_policy": source.get("wrapper", {}).get(
                "condition_scale_policy"
            ),
            "point_cloud_consumed": False,
            "gt_fit_or_metric_consumed": False,
            "passed": True,
        }
        atomic_json(destination.parent / "result.json", report)
        reports.append(report)
        print(
            f"[pose_mask_rebase] {position}/{len(inference['objects'])} "
            f"object={key} seed={seed}",
            flush=True,
        )

    manifest = {
        "format": MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "method": "native_no_vggt_pose_mask_rebased",
        "pose_mask_inference_manifest": str(inference_path),
        "pose_mask_inference_manifest_sha256": sha256_file(inference_path),
        "pose_mask_runtime_manifest": str(pose_runtime_path),
        "pose_mask_runtime_manifest_sha256": sha256_file(pose_runtime_path),
        "reference_runtime_manifest": str(reference_runtime_path),
        "reference_runtime_manifest_sha256": sha256_file(reference_runtime_path),
        "model_input_manifest": str(model_path),
        "model_input_manifest_sha256": sha256_file(model_path),
        "native_ss_checkpoint_sha256": inference["native_ss_checkpoint_sha256"],
        "native_ss_weights": inference["native_ss_weights"],
        "native_slat_checkpoint_sha256": inference["native_slat_checkpoint_sha256"],
        "native_slat_weights": inference["native_slat_weights"],
        "stock_slat_freeze_sha256": inference["stock_slat_freeze_sha256"],
        "seeds": [int(value) for value in inference["seeds"]],
        "object_count": len(inference_keys),
        "record_count": len(reports),
        "objects": reports,
        "output_frame": "reference-runtime-O",
        "coordinate_policy": (
            "observable O_posemask->W->O_reference only; no GT ICP, scale fit, "
            "translation fit, reflection, or metric-driven alignment"
        ),
        "point_cloud_consumed": False,
        "target_or_metric_consumed": False,
        "formal": formal,
        "protocol_scope": str(args.protocol_scope),
        "formal_holdout_binding": pose_runtime.get("formal_holdout_binding") if formal else None,
        "passed": len(reports) == int(inference["record_count"]),
    }
    manifest_path = output_dir / "inference_manifest.json"
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "object_count": manifest["object_count"],
                "record_count": manifest["record_count"],
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
