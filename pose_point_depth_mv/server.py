#!/usr/bin/env python3
"""Unity-compatible official ProObjaverse AR capture/reconstruction server.

This service reuses the proven recording and interactive SAM2 routes from
``trellis_point_prior_mv.server``. Its ``/generate`` endpoint validates and
materializes observable capture data, then defaults to the audited official
Native-SS step2000 + Native-SLat step25000 deployment and exports runtime-O
and AR-world meshes. The older mixed deployment remains an explicit fallback.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any


TRACKER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = TRACKER_ROOT / "pose_point_depth_mv" / "outputs" / "可视AR"
DEFAULT_GPU = os.environ.get("AR_OBJECT_GPU", os.environ.get("GPU", "4"))

# A direct ``python path/to/collect_ar_object_server.py`` launch must behave
# like the documented fully expanded deployment command before CUDA libraries
# and the legacy phone server are imported.
if __name__ == "__main__":
    if not __package__ and str(TRACKER_ROOT) not in sys.path:
        sys.path.insert(0, str(TRACKER_ROOT))
    environment_bin = str(Path(sys.executable).resolve().parent)
    inherited_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        part for part in (environment_bin, inherited_path) if part
    )
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_GPU)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("RECON_ENFORCE_INPUT_QC", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPCONV_ALGO", "native")

from flask import jsonify, request, send_file

from pose_point_depth_mv.ar_object_capture import (
    ARPointFilterConfig,
    finalize_ar_capture,
    fuse_ar_points,
    iter_ar_point_rows,
)
from pose_point_depth_mv.ar_object_reconstruction import (
    ARReconstructionConfig,
    run_ar_object_reconstruction,
)
from pose_point_depth_mv.reconstruct_real_proobjaverse_official_ss_slat import (
    DEFAULT_BRIDGE_REPORT as DEFAULT_OFFICIAL_BRIDGE_REPORT,
    DEFAULT_SLAT_CHECKPOINT as DEFAULT_OFFICIAL_SLAT_CHECKPOINT,
    DEFAULT_SS_REPORT as DEFAULT_OFFICIAL_SS_REPORT,
    DEFAULT_STOCK_FREEZE,
    RuntimeConfig as OfficialReconstructionConfig,
    run as run_official_reconstruction,
)
import trellis_point_prior_mv.server as legacy


_OUTPUT_ROOT: Path | None = None
_POINT_CONFIG = ARPointFilterConfig()
_RECONSTRUCTION_CONFIG: (
    ARReconstructionConfig | OfficialReconstructionConfig | None
) = None
_RECONSTRUCTION_DEPLOYMENT = "capture_only"
_REQUIRE_POINT_CLOUD = False
_FINALIZE_LOCK = threading.Lock()
_LATEST_PROGRESS: dict[str, Any] = {"status": "idle"}
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _configure_runtime_paths(output_root: Path) -> None:
    """Redirect every mutable legacy-server path into this collection root."""

    runtime = output_root / "runtime"
    legacy.BASE_DIR = str(runtime)
    legacy.DATA_ROOT = str(runtime / "data")
    legacy.PREVIEW_ROOT = str(runtime / "previews")
    legacy.MASK_ROOT = str(runtime / "masks")
    legacy.REVIEW_ROOT = str(runtime / "review_previews")
    legacy.OUTPUT_DIR = str(runtime / "unused_model_output")
    legacy.FLAG_DIR = str(runtime / "flags")
    legacy.DATA_DIR = legacy.DATA_ROOT
    legacy.PREVIEW_DIR = legacy.PREVIEW_ROOT

    legacy.FLAG_START_PREPROCESS = os.path.join(
        legacy.FLAG_DIR, "start_preprocess.flag"
    )
    legacy.FLAG_PREPROCESS_DONE = os.path.join(
        legacy.FLAG_DIR, "preprocess_done.json"
    )
    legacy.FLAG_START_GENERATE = os.path.join(
        legacy.FLAG_DIR, "start_generate.json"
    )
    legacy.FLAG_GENERATE_DONE = os.path.join(
        legacy.FLAG_DIR, "generate_done.flag"
    )
    legacy.FLAG_POSTPROCESS_DONE = os.path.join(
        legacy.FLAG_DIR, "postprocess_done.flag"
    )
    legacy.FLAG_CURRENT_SESSION = os.path.join(
        legacy.FLAG_DIR, "current_session.json"
    )

    legacy.current_session_id = None
    legacy.current_data_dir = legacy.DATA_ROOT
    legacy.current_preview_dir = legacy.PREVIEW_ROOT
    legacy.current_mask_dir = legacy.MASK_ROOT
    legacy.current_review_dir = legacy.REVIEW_ROOT
    legacy.current_seg_points = []
    legacy.current_seed_frames = set()
    legacy.frame_counter = 0


def _selected_frames(payload: dict[str, Any]) -> tuple[list[int], list[int], list[str], dict]:
    legacy._load_current_session()
    image_files = legacy._list_session_images()
    requested = payload.get("selected")
    if requested is None:
        requested = list(range(len(image_files)))
    selected_indices = [int(index) for index in requested]
    invalid = [index for index in selected_indices if not 0 <= index < len(image_files)]
    if invalid:
        raise ValueError(f"client selected indices are outside uploaded frames: {invalid}")
    candidate_indices = list(range(len(image_files)))
    required = int(
        _RECONSTRUCTION_CONFIG.selected_view_count
        if _RECONSTRUCTION_CONFIG is not None
        else 8
    )
    failures = []
    if len(candidate_indices) < required:
        failures.append(
            f"uploaded candidate frame count {len(candidate_indices)} < {required}"
        )
    input_qc = {
        "profile": "defer_cross_view_geometry_to_segmented_runtime_final8_v1",
        "qc_pass": not failures,
        "fail_reasons": failures,
        "warnings": [],
        "reconstruction_candidate_policy": "all_uploaded_rgb_mask_pose_frames",
        "reconstruction_candidate_count": len(candidate_indices),
        "reconstruction_candidate_indices": candidate_indices,
        "required_final_view_count": required,
        "authoritative_selection_stage": "runtime_o_segmented_all_candidates_to_final8",
        "deferred_checks": [
            "AR pose continuity segmentation",
            "per-segment mask-ray object center",
            "per-segment azimuth-balanced final-view selection",
            "final selected-view ray residual and azimuth coverage",
        ],
        "removed_redundant_gates": [
            "legacy 18-frame QC sampler",
            "whole-capture mask-ray hard gate",
            "whole-capture any-pose-jump rejection",
        ],
    }
    frame_names = [image_files[index] for index in candidate_indices]
    return selected_indices, candidate_indices, frame_names, input_qc


def _point_upload_summary() -> dict[str, Any]:
    points_path = Path(legacy.current_data_dir) / "slam_points.jsonl"
    if not points_path.is_file() or points_path.stat().st_size == 0:
        return {
            "passed": False,
            "path": str(points_path),
            "reason": "missing_or_empty_slam_points_jsonl",
            "raw_sample_count": 0,
            "temporally_supported_voxel_count": 0,
        }
    _, _, _, stats = fuse_ar_points(iter_ar_point_rows(points_path), _POINT_CONFIG)
    supported = int(stats["temporally_supported_voxel_count"])
    return {
        "passed": supported >= _POINT_CONFIG.min_object_points,
        "path": str(points_path),
        "required_temporally_supported_voxel_count": _POINT_CONFIG.min_object_points,
        **stats,
    }


def collection_input_qc():
    try:
        payload = request.get_json(silent=True) or {}
        selected, filtered, _frames, input_qc = _selected_frames(payload)
        point_upload = (
            _point_upload_summary()
            if _REQUIRE_POINT_CLOUD
            else {
                "passed": None,
                "consumed": False,
                "required": False,
                "reason": "pose_mask_runtime_bypasses_ar_point_cloud",
            }
        )
        passed = bool(input_qc.get("qc_pass")) and (
            bool(point_upload["passed"]) if _REQUIRE_POINT_CLOUD else True
        )
        reasons = list(input_qc.get("fail_reasons") or [])
        if _REQUIRE_POINT_CLOUD and not point_upload["passed"]:
            if point_upload.get("raw_sample_count", 0) == 0:
                reasons.append(
                    "未收到 ARPointCloudManager 点云；检查 Unity 中 pointCloudManager "
                    "绑定和 uploadSlamPoints 开关"
                )
            else:
                reasons.append(
                    "AR 点云时序稳定体素不足；绕物体慢速继续采集，并保证环境纹理和光照"
                )
        return jsonify(
            {
                "status": "ok" if passed else "warning",
                "message": (
                    "输入检查通过"
                    if passed
                    else "输入数据不足: " + "; ".join(reasons)
                ),
                "client_selected_indices": selected,
                "selected_indices": filtered,
                "input_qc": input_qc,
                "ar_point_upload": point_upload,
                "point_cloud_required": bool(_REQUIRE_POINT_CLOUD),
                "geometry_mode": (
                    "point_mask" if _REQUIRE_POINT_CLOUD else "pose_mask"
                ),
            }
        ), 200
    except Exception as exc:
        legacy.logging.exception("AR collection input QC failed")
        return jsonify({"status": "error", "message": str(exc)}), 500


def collection_generate():
    try:
        payload = request.get_json(silent=True) or {}
        selected, filtered, frame_names, input_qc = _selected_frames(payload)
        if legacy.arq.enforce_input_qc_from_env() and not input_qc.get(
            "qc_pass", True
        ):
            reasons = "; ".join(input_qc.get("fail_reasons") or ["unknown reason"])
            return jsonify(
                {
                    "status": "error",
                    "message": "输入帧覆盖不足，请补采后重试: " + reasons,
                    "client_selected_indices": selected,
                    "filtered_selected_indices": filtered,
                    "input_qc": input_qc,
                }
            ), 400
        if _OUTPUT_ROOT is None:
            raise RuntimeError("collection output root was not configured")
        with _FINALIZE_LOCK:
            report = finalize_ar_capture(
                session_id=str(legacy.current_session_id),
                data_dir=Path(legacy.current_data_dir),
                mask_dir=Path(legacy.current_mask_dir),
                frame_names=frame_names,
                output_root=_OUTPUT_ROOT,
                input_qc=input_qc,
                config=_POINT_CONFIG,
                require_point_cloud=bool(_REQUIRE_POINT_CLOUD),
            )
            if _RECONSTRUCTION_CONFIG is None:
                reconstruction = None
            else:
                reconstruction = _run_configured_reconstruction(
                    session_id=report["session_id"],
                    dataset_dir=Path(report["dataset_dir"]),
                    output_root=_OUTPUT_ROOT,
                )
        if reconstruction is None:
            message = f"真实采集已保存：{report['session_id']}（Pose+Mask，无点云门）"
        else:
            message = (
                f"物体重建完成：{report['session_id']}；"
                f"世界坐标 Mesh：{reconstruction['meshes']['world_glb']}；"
                f"预览拼图：{reconstruction['previews']['contact_sheet']}"
            )
        mobile_ar = (
            None
            if reconstruction is None
            else {
                "format": "yxc_unity_ar_mesh.v1",
                "mesh_url": (
                    f"/reconstruction_mesh/{report['session_id']}"
                ),
                "coordinate_frame": "unity_world",
                "placement": "world_identity",
                "display_only": True,
            }
        )
        return jsonify(
            {
                "status": "success",
                "message": message,
                "session_id": report["session_id"],
                "dataset_dir": report["dataset_dir"],
                "capture_report": str(
                    Path(report["dataset_dir"]) / "capture_report.json"
                ),
                "client_selected_indices": selected,
                "filtered_selected_indices": filtered,
                "capture": report,
                "reconstruction": reconstruction,
                "deployment_profile": _RECONSTRUCTION_DEPLOYMENT,
                "mobile_ar": mobile_ar,
            }
        ), 200
    except Exception as exc:
        legacy.logging.exception("AR collection finalization failed")
        return jsonify(
            {
                "status": "error",
                "message": _phone_error_message(exc),
                "error_detail": str(exc),
                "session_id": legacy.current_session_id,
                "session_data_dir": legacy.current_data_dir,
            }
        ), 500


def _set_latest_progress(progress: dict[str, Any]) -> None:
    global _LATEST_PROGRESS
    _LATEST_PROGRESS = dict(progress)


def _run_configured_reconstruction(
    *, session_id: str, dataset_dir: Path, output_root: Path
) -> dict[str, Any]:
    """Run exactly the deployment validated when the server was configured."""

    config = _RECONSTRUCTION_CONFIG
    if config is None:
        raise RuntimeError("reconstruction deployment is not configured")
    if isinstance(config, OfficialReconstructionConfig):
        _set_latest_progress(
            {
                "status": "running",
                "stage": "official_ss2000_slat25000",
                "session_id": session_id,
            }
        )
        try:
            result = run_official_reconstruction(
                session_id=session_id,
                dataset_dir=dataset_dir,
                output_root=output_root,
                config=config,
            )
        except Exception as error:
            _set_latest_progress(
                {
                    "status": "failed",
                    "stage": "official_ss2000_slat25000",
                    "session_id": session_id,
                    "error": repr(error),
                }
            )
            raise
        _set_latest_progress(
            {
                "status": "complete",
                "stage": "complete",
                "session_id": session_id,
                "reconstruction_report": str(
                    Path(result["run_dir"]) / "reconstruction_report.json"
                ),
                "meshes": result["meshes"],
                "previews": result["previews"],
            }
        )
        return result
    return run_ar_object_reconstruction(
        session_id=session_id,
        dataset_dir=dataset_dir,
        output_root=output_root,
        config=config,
        progress_callback=_set_latest_progress,
    )


def reconstruction_status():
    return jsonify(_LATEST_PROGRESS), 200


def _phone_error_message(error: Exception) -> str:
    detail = str(error).strip()
    if "pose-mask runtime final selected views failed input quality" in detail:
        match = re.search(r"checks=\[([^\]]*)\]", detail)
        failed = ""
        if match is not None and match.group(1).strip():
            failed = f"（{match.group(1).replace(chr(39), '').strip()}）"
        return (
            f"最终8帧质量检查未通过{failed}。请补充视角或重新采集；"
            "详细指标已记录在服务端日志。"
        )
    if len(detail) > 180:
        detail = detail[:177].rstrip() + "..."
    return "重建未完成：" + (detail or type(error).__name__)


def reconstruction_mesh(session_id: str):
    """Serve one immutable, compact Unity-world mesh to the collecting phone."""

    if _OUTPUT_ROOT is None:
        return jsonify({"status": "error", "message": "server is not configured"}), 503
    if not _SAFE_SESSION_ID.fullmatch(str(session_id)):
        return jsonify({"status": "error", "message": "invalid session id"}), 400
    mesh = (
        _OUTPUT_ROOT
        / "reconstructions"
        / str(session_id)
        / "final"
        / "reconstructed_object_world.armesh"
    )
    if not mesh.is_file():
        return jsonify(
            {
                "status": "error",
                "message": f"mobile AR mesh is unavailable for session {session_id}",
            }
        ), 404
    response = send_file(
        mesh,
        mimetype="application/octet-stream",
        as_attachment=False,
        download_name=f"{session_id}.armesh",
        conditional=True,
    )
    response.headers["X-AR-Mesh-Format"] = "yxc_unity_ar_mesh.v1"
    response.headers["X-AR-Coordinate-Frame"] = "unity_world"
    response.headers["Cache-Control"] = "private, max-age=3600, immutable"
    return response


def configure_server(
    output_root: Path,
    config: ARPointFilterConfig,
    reconstruction_config: (
        ARReconstructionConfig | OfficialReconstructionConfig | None
    ) = None,
    *,
    require_point_cloud: bool = False,
) -> None:
    global _OUTPUT_ROOT, _POINT_CONFIG, _RECONSTRUCTION_CONFIG
    global _RECONSTRUCTION_DEPLOYMENT, _REQUIRE_POINT_CLOUD
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config.validate()
    _OUTPUT_ROOT = output_root
    _POINT_CONFIG = config
    _REQUIRE_POINT_CLOUD = bool(require_point_cloud)
    if reconstruction_config is not None:
        reconstruction_config.validate()
    _RECONSTRUCTION_CONFIG = reconstruction_config
    if isinstance(reconstruction_config, OfficialReconstructionConfig):
        _RECONSTRUCTION_DEPLOYMENT = "official_ss2000_slat25000"
    elif isinstance(reconstruction_config, ARReconstructionConfig):
        _RECONSTRUCTION_DEPLOYMENT = "legacy_mixed"
    else:
        _RECONSTRUCTION_DEPLOYMENT = "capture_only"
    _configure_runtime_paths(output_root)
    legacy.app.view_functions["input_qc"] = collection_input_qc
    legacy.app.view_functions["generate"] = collection_generate
    if "reconstruction_status" not in legacy.app.view_functions:
        legacy.app.add_url_rule(
            "/reconstruction_status",
            endpoint="reconstruction_status",
            view_func=reconstruction_status,
            methods=["GET"],
        )
    if "reconstruction_mesh" not in legacy.app.view_functions:
        legacy.app.add_url_rule(
            "/reconstruction_mesh/<session_id>",
            endpoint="reconstruction_mesh",
            view_func=reconstruction_mesh,
            methods=["GET"],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture AR RGB/mask/pose/point-cloud data and reconstruct a no-VGGT Mesh"
        )
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--voxel_size_m", type=float, default=0.005)
    parser.add_argument("--min_temporal_observations", type=int, default=2)
    parser.add_argument("--min_mask_observations", type=int, default=2)
    parser.add_argument("--min_mask_support_ratio", type=float, default=0.35)
    parser.add_argument("--mask_dilation_px", type=int, default=5)
    parser.add_argument("--min_object_points", type=int, default=100)
    parser.add_argument("--max_object_points", type=int, default=50000)
    parser.add_argument("--max_point_to_mask_extent_ratio", type=float, default=2.0)
    parser.add_argument(
        "--max_ray_residual_median_over_mask_extent", type=float, default=0.20
    )
    parser.add_argument(
        "--max_ray_residual_p90_over_mask_extent", type=float, default=0.40
    )
    parser.add_argument("--min_orbit_gravity_agreement", type=float, default=0.80)
    parser.add_argument("--min_synchronized_frame_ratio", type=float, default=0.90)
    parser.add_argument(
        "--capture_only",
        action="store_true",
        help="only save the audited capture; skip SS/SLat/Mesh reconstruction",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/zjr/anaconda3/envs/reconviagen/bin/python"),
    )
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument(
        "--deployment",
        choices=("official", "legacy_mixed"),
        default="official",
        help=(
            "official (default) runs audited official Native-SS step2000 + "
            "Native-SLat step25000; legacy_mixed preserves the older deployment"
        ),
    )
    parser.add_argument(
        "--native_ss_report",
        type=Path,
        default=DEFAULT_OFFICIAL_SS_REPORT,
    )
    parser.add_argument(
        "--expected_slat_step",
        type=int,
        default=25000,
    )
    parser.add_argument(
        "--cross_deployment_bridge_report",
        type=Path,
        default=DEFAULT_OFFICIAL_BRIDGE_REPORT,
    )
    parser.add_argument(
        "--native_ss_checkpoint",
        type=Path,
        default=ARReconstructionConfig.native_ss_checkpoint,
    )
    parser.add_argument(
        "--native_slat_checkpoint",
        type=Path,
        default=None,
        help=(
            "override the SLat checkpoint; defaults to official step25000 in "
            "official mode and the frozen mixed step2000 checkpoint in legacy mode"
        ),
    )
    parser.add_argument(
        "--ss_migration_contract",
        type=Path,
        default=ARReconstructionConfig.ss_migration_contract,
    )
    parser.add_argument(
        "--slat_migration_contract",
        type=Path,
        default=ARReconstructionConfig.slat_migration_contract,
    )
    parser.add_argument(
        "--stock_slat_freeze",
        type=Path,
        default=DEFAULT_STOCK_FREEZE,
    )
    parser.add_argument(
        "--slat_backend",
        choices=("native_v2", "stock"),
        default="native_v2",
        help=(
            "native_v2 is the current phone model; stock tests Native SS support "
            "through the frozen released Stock SLat and Mesh decoder"
        ),
    )
    parser.add_argument(
        "--native_ss_cfg_strength",
        type=float,
        default=ARReconstructionConfig.native_ss_cfg_strength,
        help="Native SS CFG for the Stock-SLat test; current mixed phone value is 3.0",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--geometry_mode",
        choices=("pose_mask", "point_mask"),
        default="pose_mask",
        help="pose_mask is the current default and never gates on AR point upload",
    )
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--preview_render_frames", type=int, default=24)
    parser.add_argument("--preview_count", type=int, default=6)
    parser.add_argument("--preview_resolution", type=int, default=512)
    return parser


def build_reconstruction_config(
    args: argparse.Namespace,
) -> ARReconstructionConfig | OfficialReconstructionConfig | None:
    """Materialize one explicit deployment from parsed server arguments."""

    if args.capture_only:
        return None
    common = {
        "python": args.python,
        "gpu": str(args.gpu),
        "selected_view_count": 8,
        "geometry_mode": str(args.geometry_mode),
        "min_object_points": int(args.min_object_points),
        "min_mask_observations": int(args.min_mask_observations),
        "min_mask_support_ratio": float(args.min_mask_support_ratio),
        "seed": int(args.seed),
        "amp_dtype": str(args.amp_dtype),
        "preview_render_frames": int(args.preview_render_frames),
        "preview_count": int(args.preview_count),
        "preview_resolution": int(args.preview_resolution),
    }
    if args.deployment == "official":
        slat_checkpoint = (
            args.native_slat_checkpoint
            if args.native_slat_checkpoint is not None
            else DEFAULT_OFFICIAL_SLAT_CHECKPOINT
        )
        return OfficialReconstructionConfig(
            **common,
            view_selection_policy="object_spherical_farthest_valid_mask",
            gravity_up_w=(0.0, 1.0, 0.0) if args.geometry_mode == "pose_mask" else None,
            native_ss_report=args.native_ss_report,
            native_slat_checkpoint=slat_checkpoint,
            expected_slat_step=int(args.expected_slat_step),
            cross_deployment_bridge_report=args.cross_deployment_bridge_report,
            stock_slat_freeze=args.stock_slat_freeze,
            diagnostic_bypass_pose_mask_quality=False,
        )
    legacy_slat_checkpoint = (
        args.native_slat_checkpoint
        if args.native_slat_checkpoint is not None
        else ARReconstructionConfig.native_slat_checkpoint
    )
    return ARReconstructionConfig(
        **common,
        native_ss_checkpoint=args.native_ss_checkpoint,
        native_slat_checkpoint=legacy_slat_checkpoint,
        ss_migration_contract=args.ss_migration_contract,
        slat_migration_contract=args.slat_migration_contract,
        stock_slat_freeze=args.stock_slat_freeze,
        slat_backend=args.slat_backend,
        native_ss_cfg_strength=float(args.native_ss_cfg_strength),
    )


def main() -> None:
    args = build_parser().parse_args()
    config = ARPointFilterConfig(
        voxel_size_m=args.voxel_size_m,
        min_temporal_observations=args.min_temporal_observations,
        min_mask_observations=args.min_mask_observations,
        min_mask_support_ratio=args.min_mask_support_ratio,
        mask_dilation_px=args.mask_dilation_px,
        min_object_points=args.min_object_points,
        max_object_points=args.max_object_points,
        max_point_to_mask_extent_ratio=args.max_point_to_mask_extent_ratio,
        max_ray_residual_median_over_mask_extent=(
            args.max_ray_residual_median_over_mask_extent
        ),
        max_ray_residual_p90_over_mask_extent=(
            args.max_ray_residual_p90_over_mask_extent
        ),
        min_orbit_gravity_agreement=args.min_orbit_gravity_agreement,
        min_synchronized_frame_ratio=args.min_synchronized_frame_ratio,
    )
    reconstruction_config = build_reconstruction_config(args)
    configure_server(
        args.output_root,
        config,
        reconstruction_config,
        require_point_cloud=args.geometry_mode == "point_mask",
    )
    legacy.clean_environment()
    print(">>> [AR Object Capture] 独立真实数据采集服务启动", flush=True)
    print(f">>> 输出根目录: {_OUTPUT_ROOT}", flush=True)
    print(f">>> 点云过滤合同: {asdict(config)}", flush=True)
    if reconstruction_config is None:
        print(">>> /generate 仅固化数据（capture_only）", flush=True)
    elif isinstance(reconstruction_config, OfficialReconstructionConfig):
        print(
            ">>> /generate：采集固化 -> Pose+Mask球面筛8帧 -> DINO-only -> "
            "official Native-SS step2000 + Native-SLat step25000 -> "
            "world/AR Mesh",
            flush=True,
        )
        print(f">>> Native-SS report: {reconstruction_config.native_ss_report}", flush=True)
        print(
            f">>> Native-SLat checkpoint: {reconstruction_config.native_slat_checkpoint}",
            flush=True,
        )
        print(
            f">>> Cross-deployment bridge: "
            f"{reconstruction_config.cross_deployment_bridge_report}",
            flush=True,
        )
        print(f">>> 重建 GPU: physical cuda:{reconstruction_config.gpu}", flush=True)
    else:
        print(
            ">>> /generate 将依次运行 runtime-O、DINO-only、mixed no-VGGT "
            f"Native SS + {reconstruction_config.slat_backend} SLat、世界坐标 "
            "Mesh 导出和多视角预览渲染",
            flush=True,
        )
        print(f">>> 重建 GPU: physical cuda:{reconstruction_config.gpu}", flush=True)
    legacy.app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
