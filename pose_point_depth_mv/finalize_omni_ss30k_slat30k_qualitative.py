#!/usr/bin/env python3
"""Prepare and validate the frozen Omni Plant012 no-VGGT qualitative replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

from pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs import (
    MANIFEST_FORMAT as DINO_MANIFEST_FORMAT,
)
from pose_point_depth_mv.infer_omni_real_reconviagen import (
    MANIFEST_FORMAT as RECON_MANIFEST_FORMAT,
)
from pose_point_depth_mv.infer_real_proobjaverse_official_ss_slat import (
    MANIFEST_FORMAT as CURRENT_MANIFEST_FORMAT,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.render_runtime_o_mesh_camera_contours import (
    FORMAT as CONTOUR_FORMAT,
)
from pose_point_depth_mv.trellis_mesh_coordinate_contract import (
    validate_runtime_o_mesh_frame_contract,
)


PREPARE_FORMAT = "pose_point_depth_mv.omni_plant012_ss30k_slat30k_prepare.v1"
SUMMARY_FORMAT = "pose_point_depth_mv.omni_plant012_ss30k_slat30k_summary.v1"
EXPECTED_OBJECT_KEY = "plant:plant_012"
EXPECTED_FRAME_NAMES = [
    "00000.jpg",
    "00084.jpg",
    "00171.jpg",
    "00255.jpg",
    "00342.jpg",
    "00426.jpg",
    "00515.jpg",
    "00599.jpg",
]
EXPECTED_SOURCE_INDICES = [0, 9, 18, 27, 36, 45, 54, 63]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _single_row(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    rows = list(payload.get("objects", []))
    if len(rows) != 1:
        raise RuntimeError(f"{label} must contain exactly one object/seed")
    return rows[0]


def _validate_runtime(runtime_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = load_json(runtime_path)
    if runtime.get("passed") is not True:
        raise RuntimeError("frozen runtime input did not pass")
    row = _single_row(runtime, label="frozen runtime input")
    if (
        row.get("object_key") != EXPECTED_OBJECT_KEY
        or row.get("selected_frame_names") != EXPECTED_FRAME_NAMES
        or row.get("selected_source_view_indices") != EXPECTED_SOURCE_INDICES
        or row.get("view_selection", {}).get("fallback_used") is not False
    ):
        raise RuntimeError("frozen Plant012 eight-view identity differs")
    return runtime, row


def _validate_dino_manifest(
    manifest_path: Path, *, runtime_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = load_json(manifest_path)
    if payload.get("format") != DINO_MANIFEST_FORMAT or payload.get("passed") is not True:
        raise RuntimeError("DINO-only model input manifest did not pass")
    row = _single_row(payload, label="DINO-only model input")
    model_input = Path(row["model_input"]).expanduser().resolve(strict=True)
    if (
        row.get("object_key") != EXPECTED_OBJECT_KEY
        or row.get("vggt_model_loaded") is not False
        or row.get("vggt_model_executed") is not False
        or row.get("feature_contract", {}).get("vggt_feature_dim") != 0
        or row.get("feature_contract", {}).get("context_source") != "raw_dino_only"
        or row.get("model_input_sha256") != sha256_file(model_input)
        or payload.get("runtime_input_manifest_sha256") != sha256_file(runtime_path)
    ):
        raise RuntimeError("DINO-only/no-VGGT input identity differs")
    return payload, row


def _tree_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _verify_copied_tree(
    source: Path, destination: Path, inventory: list[dict[str, Any]]
) -> None:
    destination_files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    expected_files = {str(row["relative_path"]) for row in inventory}
    if destination_files != expected_files:
        raise RuntimeError("copied tree file set differs")
    for row in inventory:
        relative = Path(str(row["relative_path"]))
        source_path = source / relative
        destination_path = destination / relative
        if (
            source_path.stat().st_size != int(row["bytes"])
            or destination_path.stat().st_size != int(row["bytes"])
            or sha256_file(source_path) != str(row["sha256"])
            or sha256_file(destination_path) != str(row["sha256"])
        ):
            raise RuntimeError(f"copied file differs: {relative}")


def prepare(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    runtime_path = Path(args.runtime_input_manifest).expanduser().resolve(strict=True)
    dino_source = Path(args.dino_input_dir).expanduser().resolve(strict=True)
    dino_manifest_path = dino_source / "model_input_manifest.json"
    recon_source = Path(args.reconviagen_source_dir).expanduser().resolve(strict=True)
    recon_manifest_path = recon_source / "inference_manifest.json"
    _, runtime_row = _validate_runtime(runtime_path)
    _, dino_row = _validate_dino_manifest(
        dino_manifest_path, runtime_path=runtime_path
    )
    recon = load_json(recon_manifest_path)
    if recon.get("format") != RECON_MANIFEST_FORMAT or recon.get("passed") is not True:
        raise RuntimeError("source ReconViaGen manifest did not pass")
    recon_row = _single_row(recon, label="source ReconViaGen")
    recon_mesh = Path(recon_row["mesh"]).expanduser().resolve(strict=True)
    if (
        recon_row.get("object_key") != EXPECTED_OBJECT_KEY
        or int(recon_row.get("seed", -1)) != 42
        or recon_row.get("mesh_sha256") != sha256_file(recon_mesh)
    ):
        raise RuntimeError("source ReconViaGen object/seed/Mesh identity differs")

    output.mkdir(parents=True)
    dino_destination = output / "00_冻结8视图_DINO-only输入"
    recon_destination = output / "02_ReconViaGen原版输出"
    shutil.copytree(dino_source, dino_destination, copy_function=shutil.copy2)
    shutil.copytree(recon_source, recon_destination, copy_function=shutil.copy2)
    dino_inventory = _tree_inventory(dino_source)
    recon_inventory = _tree_inventory(recon_source)
    _verify_copied_tree(dino_source, dino_destination, dino_inventory)
    _verify_copied_tree(recon_source, recon_destination, recon_inventory)

    relative_mesh = recon_mesh.relative_to(recon_source)
    copied_recon_mesh = recon_destination / relative_mesh
    identity = {
        "format": PREPARE_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "object_key": EXPECTED_OBJECT_KEY,
        "runtime_input_manifest": str(runtime_path),
        "runtime_input_manifest_sha256": sha256_file(runtime_path),
        "selected_frame_names": EXPECTED_FRAME_NAMES,
        "selected_source_view_indices": EXPECTED_SOURCE_INDICES,
        "condition_sha256": runtime_row["condition_sha256"],
        "dino_only_input": {
            "source_dir": str(dino_source),
            "copied_dir": str(dino_destination),
            "source_manifest": str(dino_manifest_path),
            "source_manifest_sha256": sha256_file(dino_manifest_path),
            "copied_manifest": str(dino_destination / "model_input_manifest.json"),
            "copied_manifest_sha256": sha256_file(
                dino_destination / "model_input_manifest.json"
            ),
            "model_input_sha256": dino_row["model_input_sha256"],
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
            "files": dino_inventory,
        },
        "reconviagen_direct_copy": {
            "source_dir": str(recon_source),
            "copied_dir": str(recon_destination),
            "source_manifest": str(recon_manifest_path),
            "source_manifest_sha256": sha256_file(recon_manifest_path),
            "copied_manifest": str(recon_destination / "inference_manifest.json"),
            "copied_manifest_sha256": sha256_file(
                recon_destination / "inference_manifest.json"
            ),
            "source_mesh": str(recon_mesh),
            "copied_mesh": str(copied_recon_mesh),
            "mesh_sha256": sha256_file(recon_mesh),
            "files": recon_inventory,
        },
        "scope_guard": (
            "Exact frozen Plant012 runtime-O and DINO-only input. ReconViaGen is "
            "copied byte-for-byte from the registered with-VGGT qualitative output; "
            "it is not rerun."
        ),
    }
    atomic_json(output / "prepare_identity.json", identity)
    print(
        json.dumps(
            {
                "passed": True,
                "runtime_sha256": identity["runtime_input_manifest_sha256"],
                "dino_files": len(dino_inventory),
                "reconviagen_files": len(recon_inventory),
                "prepare_identity": str(output / "prepare_identity.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _validate_asset(path: Path, expected_sha256: str, *, label: str) -> None:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"{label} SHA256 differs")


def finalize(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).expanduser().resolve(strict=True)
    prepare_path = output / "prepare_identity.json"
    prepare_payload = load_json(prepare_path)
    if prepare_payload.get("format") != PREPARE_FORMAT or prepare_payload.get("passed") is not True:
        raise RuntimeError("prepare identity did not pass")
    runtime_path = Path(prepare_payload["runtime_input_manifest"]).resolve(strict=True)
    _, runtime_row = _validate_runtime(runtime_path)
    if sha256_file(runtime_path) != prepare_payload["runtime_input_manifest_sha256"]:
        raise RuntimeError("frozen runtime manifest changed after prepare")

    current_path = Path(args.current_manifest).expanduser().resolve(strict=True)
    current = load_json(current_path)
    if current.get("format") != CURRENT_MANIFEST_FORMAT or current.get("passed") is not True:
        raise RuntimeError("SS30K+SLat30K manifest did not pass")
    current_row = _single_row(current, label="SS30K+SLat30K")
    current_mesh = Path(current_row["mesh"]).resolve(strict=True)
    current_result = Path(current_row["result"]).resolve(strict=True)
    validate_runtime_o_mesh_frame_contract(current_row)

    native_ss_report = Path(args.native_ss_report).expanduser().resolve(strict=True)
    native_slat_checkpoint = Path(args.native_slat_checkpoint).expanduser().resolve(strict=True)
    bridge_report = Path(args.bridge_report).expanduser().resolve(strict=True)
    stock_freeze = Path(args.stock_slat_freeze).expanduser().resolve(strict=True)
    expected_ss_sha = sha256_file(native_ss_report)
    expected_slat_sha = sha256_file(native_slat_checkpoint)
    expected_bridge_sha = sha256_file(bridge_report)
    expected_stock_sha = sha256_file(stock_freeze)
    ss_binding = current.get("native_ss_deployment", {})
    slat_binding = current.get("native_slat_deployment", {})
    bridge_binding = current.get("cross_deployment_bridge", {})
    if (
        current.get("seeds") != [42]
        or current.get("object_count") != 1
        or current.get("record_count") != 1
        or current.get("output_frame") != "runtime-O"
        or current.get("vggt_model_loaded") is not False
        or current.get("vggt_model_executed") is not False
        or current.get("runtime_input_manifest_sha256")
        != prepare_payload["runtime_input_manifest_sha256"]
        or current_row.get("object_key") != EXPECTED_OBJECT_KEY
        or int(current_row.get("seed", -1)) != 42
        or current_row.get("mesh_sha256") != sha256_file(current_mesh)
        or int(ss_binding.get("checkpoint_step", -1)) != 30000
        or ss_binding.get("checkpoint_sha256") != "042a1b5467b05975584aeb571dec6ffaed5096edcc6abe4aa88600c9c9506b7f"
        or ss_binding.get("report_sha256") != expected_ss_sha
        or int(slat_binding.get("checkpoint_step", -1)) != 30000
        or slat_binding.get("checkpoint_sha256") != expected_slat_sha
        or bridge_binding.get("passed") is not True
        or bridge_binding.get("sha256") != expected_bridge_sha
        or current.get("stock_slat_freeze_sha256") != expected_stock_sha
    ):
        raise RuntimeError("SS30K+SLat30K no-VGGT deployment identity differs")

    contour_path = Path(args.contour_report).expanduser().resolve(strict=True)
    contour = load_json(contour_path)
    if (
        contour.get("format") != CONTOUR_FORMAT
        or contour.get("passed") is not True
        or contour.get("object_key") != EXPECTED_OBJECT_KEY
        or contour.get("mesh_o_sha256") != current_row["mesh_sha256"]
        or contour.get("mesh_frame_report") != str(current_result)
        or contour.get("mesh_frame_report_sha256") != sha256_file(current_result)
        or contour.get("runtime_input_manifest_sha256")
        != prepare_payload["runtime_input_manifest_sha256"]
        or contour.get("selected_frame_names") != EXPECTED_FRAME_NAMES
        or contour.get("used_physical_T_O2W") is not True
        or contour.get("used_raw_T_W2C") is not True
        or contour.get("used_T_O2C_lifting") is not False
    ):
        raise RuntimeError("SS30K+SLat30K camera contour identity differs")

    recon_copy = prepare_payload["reconviagen_direct_copy"]
    recon_mesh = Path(recon_copy["copied_mesh"]).resolve(strict=True)
    if sha256_file(recon_mesh) != recon_copy["mesh_sha256"]:
        raise RuntimeError("directly copied ReconViaGen Mesh differs")
    dino_copy = prepare_payload["dino_only_input"]
    _validate_asset(
        Path(dino_copy["copied_manifest"]).resolve(strict=True),
        dino_copy["source_manifest_sha256"],
        label="copied DINO-only manifest",
    )

    friendly_current = output / "SS30K_SLat30K_seed42" / "mesh_o.obj"
    friendly_recon = output / "ReconViaGen_original_seed42" / "mesh_reference_o.obj"
    for source, destination in (
        (current_mesh, friendly_current),
        (recon_mesh, friendly_recon),
    ):
        if destination.is_file():
            if sha256_file(destination) != sha256_file(source):
                raise RuntimeError(f"friendly Mesh differs: {destination}")
        else:
            _atomic_copy(source, destination)

    summary = {
        "format": SUMMARY_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "object_key": EXPECTED_OBJECT_KEY,
        "inference_seed": 42,
        "runtime_input_manifest": str(runtime_path),
        "runtime_input_manifest_sha256": sha256_file(runtime_path),
        "selected_frame_names": EXPECTED_FRAME_NAMES,
        "selected_source_view_indices": EXPECTED_SOURCE_INDICES,
        "selection_policy": runtime_row["view_selection"],
        "input_identity": {
            "condition_sha256": runtime_row["condition_sha256"],
            "dino_only_manifest": dino_copy["copied_manifest"],
            "dino_only_manifest_sha256": dino_copy["copied_manifest_sha256"],
            "model_input_sha256": dino_copy["model_input_sha256"],
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
        },
        "reconviagen": {
            "execution": "not rerun; byte-for-byte direct copy",
            "mesh": str(friendly_recon),
            "mesh_sha256": sha256_file(friendly_recon),
            "source_dir": recon_copy["source_dir"],
            "copied_dir": recon_copy["copied_dir"],
            "source_manifest_sha256": recon_copy["source_manifest_sha256"],
        },
        "ss30k_slat30k": {
            "endpoint": "DINO-only -> official Native-SS30K -> Native-SLat30K -> Stock Mesh decoder",
            "mesh": str(friendly_current),
            "mesh_sha256": sha256_file(friendly_current),
            "native_output_frame": current_row["output_frame"],
            "mesh_frame_contract": current_row["mesh_frame_contract"],
            "decoder_to_runtime_o_axis_transform": current_row[
                "decoder_to_runtime_o_axis_transform"
            ],
            "decoder_mesh_export_policy": current_row["decoder_mesh_export_policy"],
            "native_ss_report": str(native_ss_report),
            "native_ss_report_sha256": expected_ss_sha,
            "native_ss_checkpoint_sha256": ss_binding["checkpoint_sha256"],
            "native_ss_checkpoint_step": ss_binding["checkpoint_step"],
            "native_slat_checkpoint": str(native_slat_checkpoint),
            "native_slat_checkpoint_sha256": expected_slat_sha,
            "native_slat_checkpoint_step": slat_binding["checkpoint_step"],
            "source_manifest": str(current_path),
            "source_manifest_sha256": sha256_file(current_path),
            "camera_contour_report": str(contour_path),
            "camera_contour_report_sha256": sha256_file(contour_path),
            "camera_contour_overview": contour["overview"],
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
        },
        "coordinate_note": (
            "The SS30K+SLat30K decoder vertices already share the native "
            "sparse-grid/runtime-O axes. The Mesh is exported with "
            "transform_pose=False and no presentation-axis rotation. Contours "
            "use physical T_O2W, raw T_W2C, K and distortion."
        ),
        "scope_guard": (
            "One frozen real Omni sample for qualitative comparison. ReconViaGen "
            "is the exact previously generated result; only the no-VGGT "
            "SS30K+SLat30K endpoint is newly executed."
        ),
    }
    atomic_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# Omni Plant012 冻结8视图：ReconViaGen 与 no-VGGT SS30K+SLat30K\n\n"
        "- 与旧 VSS2k+V-SLat15k 实验严格复用同一 runtime-O、8张输入及相机。\n"
        "- `00_冻结8视图_DINO-only输入`：原 DINO-only 输入的逐文件精确副本；VGGT未加载、未执行。\n"
        "- `02_ReconViaGen原版输出`：旧实验结果的逐文件直接副本，未重新推理。\n"
        "- `SS30K_SLat30K_seed42`：no-VGGT SS30K+SLat30K 的 runtime-O Mesh。\n"
        "- `SS30K_SLat30K_相机轮廓`：使用物理 T_O2W、原始 T_W2C、K及畸变投影的青色轮廓。\n"
        "- decoder 顶点与 sparse-grid/runtime-O 原生同轴；使用 `transform_pose=False`，不施加展示用轴旋转。\n"
        "- 原图轮廓没有使用 projective-normalized T_O2C_lifting。\n\n"
        "这是单个真实样本的定性结果，不构成新的 held-out 定量结论。\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "summary": str(output / "summary.json"),
                "current_mesh": str(friendly_current),
                "reconviagen_mesh": str(friendly_recon),
                "contour_overview": contour["overview"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--runtime_input_manifest", required=True)
    p.add_argument("--dino_input_dir", required=True)
    p.add_argument("--reconviagen_source_dir", required=True)
    f = commands.add_parser("finalize")
    f.add_argument("--output_dir", required=True)
    f.add_argument("--current_manifest", required=True)
    f.add_argument("--contour_report", required=True)
    f.add_argument("--native_ss_report", required=True)
    f.add_argument("--native_slat_checkpoint", required=True)
    f.add_argument("--bridge_report", required=True)
    f.add_argument("--stock_slat_freeze", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    {"prepare": prepare, "finalize": finalize}[args.command](args)


if __name__ == "__main__":
    main()
