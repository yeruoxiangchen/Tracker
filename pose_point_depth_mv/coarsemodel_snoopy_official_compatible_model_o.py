#!/usr/bin/env python3
"""Rebuild Snoopy with a Z-up, official-compatible internal model-O.

The source pose-mask runtime-O is retained as the physical/world placement.
Only the coordinate frame presented to SS30K/SLat30K is changed:

    model +X = source runtime +X
    model +Y = source runtime -Z
    model +Z = source runtime +Y (estimated up)

The transform is a proper rotation.  Center, scale, physical cameras, RGB,
masks, selected views and cached DINO features remain bit-identical.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from pose_point_depth_mv.coarsemodel_fixed_colmap_ss30k_slat30k import (
    _asset,
    _copy_verified,
    _only_object,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs import (
    MANIFEST_FORMAT as MODEL_MANIFEST_FORMAT,
    OBJECT_FORMAT as MODEL_OBJECT_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    atomic_torch_save,
    canonical_sha256,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.real_object_canonicalization import (
    array_sha256,
    canonical_json_sha256,
)
from pose_point_depth_mv.trellis_mesh_coordinate_contract import (
    validate_runtime_o_mesh_frame_contract,
)


TRACKER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    TRACKER_ROOT
    / "pose_point_depth_mv/outputs2/"
    "CoarseModel_snoopy_指定8帧_COLMAPPose_"
    "ReconViaGen_vs_SS30K_SLat30K_runtimeO轮廓_20260819_v1"
)
RUNTIME_FORMAT = "pose_point_depth_mv.official_compatible_model_o_runtime.v1"
MODEL_REUSE_FORMAT = "pose_point_depth_mv.official_compatible_model_o_dino_reuse.v1"
FINAL_FORMAT = "pose_point_depth_mv.coarsemodel_snoopy_official_compatible_model_o.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_to_source_runtime_o() -> np.ndarray:
    """Return Q where p_source_runtime = Q @ p_model."""

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        ),
        dtype=np.float64,
    )
    if not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3)):
        raise RuntimeError("model-O conversion is not orthonormal")
    if not np.isclose(np.linalg.det(result[:3, :3]), 1.0):
        raise RuntimeError("model-O conversion is not a proper rotation")
    return result


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _scale_of(transform: np.ndarray) -> float:
    linear = np.asarray(transform[:3, :3], dtype=np.float64)
    singular = np.linalg.svd(linear, compute_uv=False)
    if not np.allclose(singular, singular.mean(), rtol=1.0e-7, atol=1.0e-9):
        raise RuntimeError(f"runtime-O transform is not a similarity: {singular}")
    return float(singular.mean())


def _transform_runtime_arrays(
    source: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = {
        "T_O2W",
        "T_W2O",
        "T_O2C",
        "T_O2C_lifting",
        "T_C2O",
        "P_O",
    }
    missing = sorted(required - set(source))
    if missing:
        raise RuntimeError(f"source runtime cache lacks fields: {missing}")
    q = model_to_source_runtime_o()
    old_o2w = np.asarray(source["T_O2W"], dtype=np.float64)
    old_o2c = np.asarray(source["T_O2C"], dtype=np.float64)
    old_lifting = np.asarray(source["T_O2C_lifting"], dtype=np.float64)
    new_o2w = old_o2w @ q
    new_w2o = np.linalg.inv(new_o2w)
    new_o2c = old_o2c @ q[None]
    new_lifting = old_lifting @ q[None]
    new_c2o = np.linalg.inv(new_o2c)
    old_points = np.asarray(source["P_O"])
    new_points = (old_points @ q[:3, :3]).astype(old_points.dtype, copy=False)

    transformed = {name: np.asarray(value) for name, value in source.items()}
    transformed.update(
        {
            "T_O2W": new_o2w,
            "T_W2O": new_w2o,
            "T_O2C": new_o2c,
            "T_O2C_lifting": new_lifting,
            "T_C2O": new_c2o,
            "P_O": new_points,
        }
    )
    old_scale, new_scale = _scale_of(old_o2w), _scale_of(new_o2w)
    old_rotation = old_o2w[:3, :3] / old_scale
    new_rotation = new_o2w[:3, :3] / new_scale
    checks = {
        "proper_rotation": bool(np.isclose(np.linalg.det(q[:3, :3]), 1.0)),
        "center_unchanged": bool(
            np.allclose(old_o2w[:3, 3], new_o2w[:3, 3], atol=1.0e-12)
        ),
        "scale_unchanged": bool(np.isclose(old_scale, new_scale, atol=1.0e-12)),
        "model_z_equals_source_runtime_y": bool(
            np.allclose(new_rotation[:, 2], old_rotation[:, 1], atol=1.0e-12)
        ),
        "model_x_equals_source_runtime_x": bool(
            np.allclose(new_rotation[:, 0], old_rotation[:, 0], atol=1.0e-12)
        ),
        "physical_camera_chain_exact": bool(
            np.allclose(new_o2c, old_o2c @ q[None], atol=1.0e-12)
        ),
        "lifting_camera_chain_exact": bool(
            np.allclose(new_lifting, old_lifting @ q[None], atol=1.0e-12)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"model-O conversion checks failed: {checks}")
    audit = {
        "format": RUNTIME_FORMAT,
        "Q_model_O_to_source_runtime_O": q.tolist(),
        "axis_mapping": {
            "model_plus_x": "source_runtime_plus_x",
            "model_plus_y": "source_runtime_minus_z",
            "model_plus_z": "source_runtime_plus_y_estimated_up",
        },
        "determinant": float(np.linalg.det(q[:3, :3])),
        "source_scale_O2W": old_scale,
        "model_scale_O2W": new_scale,
        "center_W": new_o2w[:3, 3].tolist(),
        "checks": checks,
    }
    return transformed, audit


def _runtime_frame_record(
    source: dict[str, Any], arrays: dict[str, np.ndarray], audit: dict[str, Any]
) -> dict[str, Any]:
    result = deepcopy(source)
    contract = deepcopy(result["contract"])
    contract["format"] = RUNTIME_FORMAT
    contract["source_axis_rule"] = contract.get("axis_rule")
    contract["axis_rule"] = (
        "official-compatible model-O: +Z=source runtime estimated up; "
        "+X=source runtime +X; +Y=source runtime -Z"
    )
    contract["model_o_conversion"] = audit
    result["contract"] = contract
    result["contract_sha256"] = canonical_json_sha256(contract)
    result["T_O2W"] = arrays["T_O2W"].tolist()
    result["T_W2O"] = arrays["T_W2O"].tolist()
    result["T_O2C_sha256"] = array_sha256(arrays["T_O2C"])
    result["T_O2C_lifting_sha256"] = array_sha256(arrays["T_O2C_lifting"])
    result["T_C2O_sha256"] = array_sha256(arrays["T_C2O"])
    result["P_O_sha256"] = array_sha256(arrays["P_O"])
    stats = deepcopy(result.get("stats") or {})
    stats["official_compatible_model_o"] = audit
    result["stats"] = stats
    result.pop("frame_sha256", None)
    result["frame_sha256"] = canonical_json_sha256(result)
    return result


def build_runtime(source_manifest: Path, output: Path) -> None:
    source_manifest = source_manifest.expanduser().resolve(strict=True)
    source = load_json(source_manifest)
    if source.get("format") != RUNTIME_MANIFEST_FORMAT or source.get("passed") is not True:
        raise RuntimeError("source runtime manifest is not eligible")
    if output.exists():
        report = output / "runtime_input_manifest.json"
        if report.is_file() and load_json(report).get("passed") is True:
            print(json.dumps({"passed": True, "reused": True, "manifest": str(report)}, indent=2))
            return
        raise RuntimeError(f"partial model-O runtime output exists: {output}")

    rows = []
    audits = []
    for source_row in source["objects"]:
        source_cache = Path(source_row["cache_npz"]).expanduser().resolve(strict=True)
        source_condition_path = Path(source_row["condition_record"]).expanduser().resolve(strict=True)
        if sha256_file(source_cache) == "" or sha256_file(source_condition_path) == "":
            raise RuntimeError("source runtime binding is empty")
        with np.load(source_cache, allow_pickle=False) as payload:
            source_arrays = {name: np.asarray(payload[name]) for name in payload.files}
        arrays, audit = _transform_runtime_arrays(source_arrays)
        object_dir = output / "objects" / source_row["category"] / source_row["object_id"]
        cache_path = object_dir / "runtime_input_cache.npz"
        condition_path = object_dir / "condition_record.json"
        report_path = object_dir / "report.json"
        object_dir.mkdir(parents=True, exist_ok=False)
        _atomic_npz(cache_path, arrays)

        condition = load_json(source_condition_path)
        condition["format"] = RUNTIME_FORMAT
        condition["source_condition_record"] = str(source_condition_path)
        condition["source_condition_record_sha256"] = sha256_file(source_condition_path)
        condition["runtime_frame"] = _runtime_frame_record(
            condition["runtime_frame"], arrays, audit
        )
        condition["T_O2C_sha256"] = array_sha256(arrays["T_O2C"])
        condition["T_O2C_lifting_sha256"] = array_sha256(arrays["T_O2C_lifting"])
        condition["P_O_sha256"] = array_sha256(arrays["P_O"])
        condition["condition_scope"] = (
            "same observable pose+mask input; official-compatible Z-up model-O "
            "is a proper re-expression of the source runtime-O"
        )
        condition.pop("condition_sha256", None)
        condition["condition_sha256"] = canonical_json_sha256(condition)
        atomic_json(condition_path, condition)

        row = deepcopy(source_row)
        row.update(
            {
                "format": RUNTIME_FORMAT,
                "created_at_utc": utc_now(),
                "source_runtime_input_manifest": str(source_manifest),
                "source_runtime_input_manifest_sha256": sha256_file(source_manifest),
                "source_runtime_cache": str(source_cache),
                "source_runtime_cache_sha256": sha256_file(source_cache),
                "input_frontend_format": RUNTIME_FORMAT,
                "cache_npz": str(cache_path),
                "condition_record": str(condition_path),
                "condition_sha256": condition["condition_sha256"],
                "T_O2C_sha256": array_sha256(arrays["T_O2C"]),
                "T_O2C_lifting_sha256": array_sha256(arrays["T_O2C_lifting"]),
                "model_o_conversion": audit,
                "scope_guard": (
                    "Source RGB/mask/COLMAP pose, view selection, center and scale are "
                    "unchanged; only a proper source-runtime-O to Z-up model-O axis "
                    "conversion is applied. No target or reconstructed Mesh is consumed."
                ),
                "passed": True,
            }
        )
        row["runtime_frame_stats"] = deepcopy(row.get("runtime_frame_stats") or {})
        row["runtime_frame_stats"]["official_compatible_model_o"] = audit
        atomic_json(report_path, row)
        rows.append(row)
        audits.append(audit)

    build_config = deepcopy(source.get("build_config") or {})
    build_config.update(
        {
            "input_frontend_format": RUNTIME_FORMAT,
            "source_runtime_input_manifest": str(source_manifest),
            "source_runtime_input_manifest_sha256": sha256_file(source_manifest),
            "model_o_conversion": audits[0] if len(audits) == 1 else audits,
        }
    )
    build_hash = canonical_sha256(build_config)
    for row in rows:
        row["build_config_sha256"] = build_hash
        atomic_json(Path(row["cache_npz"]).parent / "report.json", row)
    manifest = deepcopy(source)
    manifest.update(
        {
            "created_at_utc": utc_now(),
            "source_runtime_input_manifest": str(source_manifest),
            "source_runtime_input_manifest_sha256": sha256_file(source_manifest),
            "build_config": build_config,
            "build_config_sha256": build_hash,
            "objects": rows,
            "completed_object_count": len(rows),
            "failures": [],
            "passed": True,
            "scope_guard": (
                "Official-compatible model-O derivative; center, scale, images, masks, "
                "physical cameras and selected views are unchanged."
            ),
        }
    )
    manifest_path = output / "runtime_input_manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({"passed": True, "objects": len(rows), "manifest": str(manifest_path)}, indent=2))


def build_model_input(
    source_manifest: Path, runtime_manifest: Path, output: Path
) -> None:
    source_manifest = source_manifest.expanduser().resolve(strict=True)
    runtime_manifest = runtime_manifest.expanduser().resolve(strict=True)
    source = load_json(source_manifest)
    runtime = load_json(runtime_manifest)
    if source.get("format") != MODEL_MANIFEST_FORMAT or source.get("passed") is not True:
        raise RuntimeError("source DINO model-input manifest is not eligible")
    if runtime.get("format") != RUNTIME_MANIFEST_FORMAT or runtime.get("passed") is not True:
        raise RuntimeError("model-O runtime manifest is not eligible")
    if output.exists():
        report = output / "model_input_manifest.json"
        if report.is_file() and load_json(report).get("passed") is True:
            print(json.dumps({"passed": True, "reused": True, "manifest": str(report)}, indent=2))
            return
        raise RuntimeError(f"partial model-O model input exists: {output}")

    source_by_key = {str(row["object_key"]): row for row in source["objects"]}
    reports = []
    for runtime_row in runtime["objects"]:
        key = str(runtime_row["object_key"])
        source_row = source_by_key[key]
        source_payload = Path(source_row["model_input"]).expanduser().resolve(strict=True)
        if sha256_file(source_payload) != str(source_row["model_input_sha256"]):
            raise RuntimeError("source DINO model-input hash differs")
        payload = torch.load(source_payload, map_location="cpu")
        cache = Path(runtime_row["cache_npz"]).expanduser().resolve(strict=True)
        with np.load(cache, allow_pickle=False) as geometry:
            intrinsics = np.asarray(geometry["K_feature"])
            extrinsics = np.asarray(geometry["T_O2C_lifting"])
            points_o = np.asarray(geometry["P_O"])
        payload.update(
            {
                "condition_sha256": str(runtime_row["condition_sha256"]),
                "runtime_cache": str(cache),
                "runtime_cache_sha256": sha256_file(cache),
                "intrinsics": torch.from_numpy(intrinsics),
                "extrinsics": torch.from_numpy(extrinsics.astype(np.float32)),
                "points_o": torch.from_numpy(points_o.astype(np.float32)),
                "extrinsics_source": "official_compatible_model_o.T_O2C_lifting",
            }
        )
        destination = output / "objects" / runtime_row["category"] / runtime_row["object_id"]
        destination.mkdir(parents=True, exist_ok=False)
        payload_path = destination / "dino_only_model_input.pt"
        atomic_torch_save(payload_path, payload)
        report = deepcopy(source_row)
        report.update(
            {
                "format": MODEL_OBJECT_FORMAT,
                "created_at_utc": utc_now(),
                "runtime_cache": str(cache),
                "runtime_cache_sha256": sha256_file(cache),
                "condition_sha256": str(runtime_row["condition_sha256"]),
                "model_input": str(payload_path),
                "model_input_sha256": sha256_file(payload_path),
                "source_model_input": str(source_payload),
                "source_model_input_sha256": sha256_file(source_payload),
                "visual_features_reused_bit_exact": True,
                "geometry_fields_replaced": ["extrinsics", "runtime_cache", "points_o"],
                "target_or_mesh_consumed": False,
                "scope_guard": (
                    "DINO tensors are reused bit-exactly; only the audited Z-up model-O "
                    "lifting extrinsics/runtime cache replace the source geometry."
                ),
                "passed": True,
            }
        )
        atomic_json(destination / "report.json", report)
        reports.append(report)
    manifest = {
        "format": MODEL_MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "runtime_input_manifest": str(runtime_manifest),
        "runtime_input_manifest_sha256": sha256_file(runtime_manifest),
        "source_model_input_manifest": str(source_manifest),
        "source_model_input_manifest_sha256": sha256_file(source_manifest),
        "config": {
            "format": MODEL_REUSE_FORMAT,
            "visual_features": "bit-exact reuse",
            "geometry": "official-compatible Z-up model-O",
        },
        "config_sha256": canonical_sha256(
            {
                "format": MODEL_REUSE_FORMAT,
                "runtime_input_manifest_sha256": sha256_file(runtime_manifest),
                "source_model_input_manifest_sha256": sha256_file(source_manifest),
            }
        ),
        "selected_object_count": len(reports),
        "completed_object_count": len(reports),
        "reused_objects": [],
        "objects": reports,
        "failures": [],
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "training_ready": False,
        "scope_guard": "Bit-exact DINO reuse with audited model-O geometry replacement.",
        "passed": True,
    }
    manifest_path = output / "model_input_manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({"passed": True, "objects": len(reports), "manifest": str(manifest_path)}, indent=2))


def _paired_overview(old_path: Path, new_path: Path, destination: Path) -> None:
    old = Image.open(old_path).convert("RGB")
    new = Image.open(new_path).convert("RGB")
    if old.size != new.size:
        new = new.resize(old.size, Image.Resampling.LANCZOS)
    header = 42
    sheet = Image.new("RGB", (old.width * 2, old.height + header), (20, 20, 20))
    sheet.paste(old, (0, header))
    sheet.paste(new, (old.width, header))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 13), "Old runtime-O (+Y up)", fill=(238, 238, 238))
    draw.text((old.width + 12, 13), "Official-compatible model-O (+Z up)", fill=(238, 238, 238))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    sheet.save(temporary, format="PNG")
    os.replace(temporary, destination)


def finalize(root: Path) -> None:
    root = root.expanduser().resolve(strict=True)
    output = root / "12_official_compatible_model_o结果汇总"
    final_report = root / "official_compatible_model_o_report.json"
    if final_report.is_file() and load_json(final_report).get("passed") is True:
        print(json.dumps({"passed": True, "reused": True, "report": str(final_report)}, indent=2))
        return
    if output.exists() or final_report.exists():
        raise RuntimeError("partial official-compatible model-O final output exists")
    old_report = load_json(root / "report.json")
    runtime_path = root / "objects/snoopy/08_official_compatible_model_o_runtime/runtime_input_manifest.json"
    model_path = root / "objects/snoopy/09_official_compatible_model_o_dino_input/model_input_manifest.json"
    infer_path = root / "objects/snoopy/10_official_compatible_model_o_ss30k_slat30k/inference_manifest.json"
    contour_path = root / "objects/snoopy/11_official_compatible_model_o_camera_contours/report.json"
    for path in (runtime_path, model_path, infer_path, contour_path):
        if load_json(path).get("passed") is not True:
            raise RuntimeError(f"incomplete model-O artifact: {path}")
    runtime_row = _only_object(load_json(runtime_path), runtime_path)
    model_row = _only_object(load_json(model_path), model_path)
    infer_row = _only_object(load_json(infer_path), infer_path)
    contour = load_json(contour_path)
    validate_runtime_o_mesh_frame_contract(infer_row)
    new_mesh = Path(infer_row["mesh"]).resolve(strict=True)
    if contour.get("mesh_o_sha256") != sha256_file(new_mesh):
        raise RuntimeError("new contour/Mesh binding differs")
    if contour.get("runtime_input_manifest_sha256") != sha256_file(runtime_path):
        raise RuntimeError("new contour/runtime binding differs")
    if model_row.get("condition_sha256") != runtime_row.get("condition_sha256"):
        raise RuntimeError("model input/runtime condition binding differs")
    old_mesh = Path(old_report["current_mesh"]["path"]).resolve(strict=True)
    recon_mesh = Path(old_report["reconviagen_mesh"]["path"]).resolve(strict=True)
    old_contour = Path(old_report["contour_overview"]["path"]).resolve(strict=True)
    new_contour = Path(contour["overview"]).resolve(strict=True)
    output.mkdir(parents=True)
    files = {
        "old_runtime_o_mesh": output / "01_旧runtimeO_SS30K_SLat30K.obj",
        "official_compatible_model_o_mesh": output / "02_official兼容modelO_SS30K_SLat30K.obj",
        "reconviagen_mesh": output / "03_ReconViaGen原版.obj",
        "old_runtime_o_contours": output / "04_旧runtimeO轮廓总览.png",
        "official_compatible_model_o_contours": output / "05_official兼容modelO轮廓总览.png",
        "paired_contours": output / "06_旧runtimeO_vs_official兼容modelO_轮廓并排.png",
    }
    _copy_verified(old_mesh, files["old_runtime_o_mesh"])
    _copy_verified(new_mesh, files["official_compatible_model_o_mesh"])
    _copy_verified(recon_mesh, files["reconviagen_mesh"])
    _copy_verified(old_contour, files["old_runtime_o_contours"])
    _copy_verified(new_contour, files["official_compatible_model_o_contours"])
    _paired_overview(old_contour, new_contour, files["paired_contours"])
    conversion = runtime_row["model_o_conversion"]
    report = {
        "format": FINAL_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "formal": False,
        "object": "snoopy",
        "view_count": 8,
        "source_experiment_report": _asset(root / "report.json"),
        "runtime_input_manifest": _asset(runtime_path),
        "model_input_manifest": _asset(model_path),
        "inference_manifest": _asset(infer_path),
        "contour_report": _asset(contour_path),
        "model_o_conversion": conversion,
        "invariants": {
            "same_eight_rgb": True,
            "same_eight_masks": True,
            "same_colmap_T_W2C": True,
            "same_center_W": True,
            "same_scale_O2W": True,
            "same_seed": 42,
            "same_ss30k_slat30k": True,
            "same_dino_features_bit_exact": True,
            "only_model_o_axes_changed": True,
        },
        "new_mesh_structure": infer_row["structure"],
        "outputs": {name: _asset(path) for name, path in files.items()},
        "scope_guard": (
            "Single real Snoopy qualitative A/B. The new branch changes only the "
            "internal O-axis convention and maps the decoded Mesh through the new "
            "T_O2W for projection; it is not a GT geometry benchmark."
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_json(final_report, report)
    print(json.dumps({"passed": True, "report": str(final_report), "output": str(output)}, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runtime = sub.add_parser("build-runtime")
    runtime.add_argument("--source_manifest", type=Path, required=True)
    runtime.add_argument("--output_dir", type=Path, required=True)
    model = sub.add_parser("build-model-input")
    model.add_argument("--source_manifest", type=Path, required=True)
    model.add_argument("--runtime_manifest", type=Path, required=True)
    model.add_argument("--output_dir", type=Path, required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "build-runtime":
        build_runtime(args.source_manifest, args.output_dir)
    elif args.command == "build-model-input":
        build_model_input(args.source_manifest, args.runtime_manifest, args.output_dir)
    else:
        finalize(args.root)


if __name__ == "__main__":
    main()
