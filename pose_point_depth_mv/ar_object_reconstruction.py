#!/usr/bin/env python3
"""Resumable single-object no-VGGT reconstruction orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

import trimesh

from pose_point_depth_mv.ar_object_capture import utc_now, write_json
from pose_point_depth_mv.ar_mobile_overlay import build_mobile_overlay_mesh


RECONSTRUCTION_FORMAT = "pose_point_depth_mv.ar_object_reconstruction.v1"
FRAME_SELECTION_FORMAT = "pose_point_depth_mv.ar_object_phone_frame_selection.v1"


@dataclass(frozen=True)
class ARReconstructionConfig:
    python: Path = Path("/home/zjr/anaconda3/envs/reconviagen/bin/python")
    gpu: str = "4"
    native_ss_checkpoint: Path = Path(
        "/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/"
        "ss_mixed_step2000_seed42_1gpu_v1/checkpoints/step_002000.pt"
    )
    native_slat_checkpoint: Path = Path(
        "/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/"
        "slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_002000.pt"
    )
    ss_migration_contract: Path = Path(
        "/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/"
        "contracts/ss_real_full_ema_v1.json"
    )
    slat_migration_contract: Path = Path(
        "/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/"
        "contracts/slat_real_full_ema_v1.json"
    )
    stock_slat_freeze: Path = Path(
        "/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/"
        "stock_slat_freeze_v2.json"
    )
    slat_backend: str = "native_v2"
    native_ss_cfg_strength: float = 3.0
    selected_view_count: int = 8
    geometry_mode: str = "pose_mask"
    min_object_points: int = 100
    min_mask_observations: int = 2
    min_mask_support_ratio: float = 0.35
    view_selection_policy: str = "object_spherical_farthest_valid_mask"
    gravity_up_w: tuple[float, float, float] | None = (0.0, 1.0, 0.0)
    source_frame_names: tuple[str, ...] | None = None
    seed: int = 42
    amp_dtype: str = "bf16"
    preview_render_frames: int = 24
    preview_count: int = 6
    preview_resolution: int = 512
    diagnostic_bypass_pose_mask_quality: bool = False

    def validate(self) -> None:
        required = {
            "python": self.python,
            "native_ss_checkpoint": self.native_ss_checkpoint,
            "ss_migration_contract": self.ss_migration_contract,
            "stock_slat_freeze": self.stock_slat_freeze,
        }
        if self.slat_backend == "native_v2":
            required.update(
                {
                    "native_slat_checkpoint": self.native_slat_checkpoint,
                    "slat_migration_contract": self.slat_migration_contract,
                }
            )
        missing = [f"{name}={path}" for name, path in required.items() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError("missing reconstruction deployment files: " + "; ".join(missing))
        gpu_parts = [part.strip() for part in str(self.gpu).split(",") if part.strip()]
        if len(gpu_parts) != 1 or not gpu_parts[0].isdigit():
            raise ValueError("AR reconstruction requires exactly one physical GPU index")
        if self.selected_view_count < 8:
            raise ValueError("selected_view_count must be at least 8")
        if self.geometry_mode not in {"pose_mask", "point_mask"}:
            raise ValueError("geometry_mode must be pose_mask or point_mask")
        if self.slat_backend not in {"native_v2", "stock"}:
            raise ValueError("slat_backend must be native_v2 or stock")
        if self.native_ss_cfg_strength <= 0.0:
            raise ValueError("native_ss_cfg_strength must be positive")
        if self.min_object_points < 1:
            raise ValueError("min_object_points must be positive")
        if self.min_mask_observations < 1:
            raise ValueError("min_mask_observations must be positive")
        if not 0.0 <= self.min_mask_support_ratio <= 1.0:
            raise ValueError("min_mask_support_ratio must be in [0,1]")
        if self.view_selection_policy not in {
            "lexical_even",
            "lexical_even_valid_mask_fallback",
            "object_azimuth_balanced_valid_mask",
            "object_spherical_farthest_valid_mask",
        }:
            raise ValueError(f"unsupported view_selection_policy={self.view_selection_policy!r}")
        if self.gravity_up_w is not None:
            if len(self.gravity_up_w) != 3:
                raise ValueError("gravity_up_w must contain exactly three values")
            if sum(float(value) ** 2 for value in self.gravity_up_w) <= 1.0e-12:
                raise ValueError("gravity_up_w must be nonzero")
        if self.source_frame_names is not None:
            if len(self.source_frame_names) != self.selected_view_count:
                raise ValueError(
                    "source_frame_names must exactly match selected_view_count"
                )
            if len(set(self.source_frame_names)) != len(self.source_frame_names):
                raise ValueError("source_frame_names must be unique")
            if any(not str(name).strip() for name in self.source_frame_names):
                raise ValueError("source_frame_names must be nonempty")
        if self.amp_dtype not in {"bf16", "fp16", "none"}:
            raise ValueError("amp_dtype must be bf16, fp16, or none")
        if self.preview_count < 1:
            raise ValueError("preview_count must be positive")
        if self.preview_render_frames < self.preview_count:
            raise ValueError("preview_render_frames must be at least preview_count")
        if self.preview_resolution < 64:
            raise ValueError("preview_resolution must be at least 64")


@dataclass(frozen=True)
class ReconstructionPaths:
    run_dir: Path
    raw_dir: Path
    runtime_dir: Path
    model_input_dir: Path
    inference_dir: Path
    bundle_dir: Path
    final_dir: Path

    @classmethod
    def build(cls, output_root: Path, session_id: str) -> "ReconstructionPaths":
        run_dir = output_root / "reconstructions" / session_id
        return cls(
            run_dir=run_dir,
            raw_dir=run_dir / "01_raw_cache",
            runtime_dir=run_dir / "02_runtime_o",
            model_input_dir=run_dir / "03_dino_only_input",
            inference_dir=run_dir / "04_no_vggt_inference",
            bundle_dir=run_dir / "05_world_mesh_bundle",
            final_dir=run_dir / "final",
        )


def reconstruction_commands(
    dataset_dir: Path,
    paths: ReconstructionPaths,
    config: ARReconstructionConfig,
) -> list[tuple[str, list[str], Path]]:
    python = str(config.python)
    raw_manifest = paths.raw_dir / "raw_cache_report.json"
    runtime_manifest = paths.runtime_dir / "runtime_input_manifest.json"
    model_manifest = paths.model_input_dir / "model_input_manifest.json"
    inference_manifest = paths.inference_dir / "inference_manifest.json"
    raw_command = [
        python,
        "-u",
        "-m",
        "pose_point_depth_mv.dataset_tools.prepare_coarsemodel_real_raw_cache",
        "--dataset",
        str(dataset_dir),
        "--output_dir",
        str(paths.raw_dir),
        "--min_registered_pairs",
        str(config.selected_view_count),
    ]
    if config.geometry_mode == "pose_mask":
        raw_command.append("--allow_empty_points")
    if config.source_frame_names is not None:
        for frame_name in config.source_frame_names:
            raw_command.extend(["--frame_name", str(frame_name)])
    raw_command.append("--resume")
    runtime_command = [
        python,
        "-u",
        "-m",
        "pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs",
        "--raw_cache_report",
        str(raw_manifest),
        "--output_dir",
        str(paths.runtime_dir),
        "--selected_view_count",
        str(config.selected_view_count),
        "--view_selection_policy",
        config.view_selection_policy,
        "--geometry_mode",
        config.geometry_mode,
        "--min_completed_objects",
        "1",
    ]
    if config.geometry_mode == "point_mask":
        runtime_command.extend(
            [
                "--min_object_points",
                str(config.min_object_points),
                "--min_mask_observations",
                str(config.min_mask_observations),
                "--min_mask_support_ratio",
                str(config.min_mask_support_ratio),
            ]
        )
    if config.gravity_up_w is not None:
        runtime_command.extend(
            ["--gravity_up_w", *(f"{float(value):g}" for value in config.gravity_up_w)]
        )
    runtime_command.append("--resume")
    if config.slat_backend == "native_v2":
        inference_name = "no_vggt_ss_slat_mesh"
        inference_command = [
            python,
            "-u",
            "-m",
            "pose_point_depth_mv.infer_omni_real_native_no_vggt_mixed",
            "--model_input_manifest",
            str(model_manifest),
            "--native_ss_checkpoint",
            str(config.native_ss_checkpoint),
            "--native_slat_checkpoint",
            str(config.native_slat_checkpoint),
            "--ss_migration_contract",
            str(config.ss_migration_contract),
            "--slat_migration_contract",
            str(config.slat_migration_contract),
            "--stock_slat_freeze",
            str(config.stock_slat_freeze),
            "--output_dir",
            str(paths.inference_dir),
            "--seeds",
            str(config.seed),
            "--weights",
            "ema",
            "--amp_dtype",
            config.amp_dtype,
            "--device",
            "cuda",
        ]
    else:
        inference_name = "no_vggt_native_ss_stock_slat_mesh"
        inference_command = [
            python,
            "-u",
            "-m",
            "pose_point_depth_mv.infer_omni_real_native_ss_stock_slat_no_vggt",
            "--model_input_manifest",
            str(model_manifest),
            "--native_ss_checkpoint",
            str(config.native_ss_checkpoint),
            "--native_ss_cfg_strength",
            str(config.native_ss_cfg_strength),
            "--ss_migration_contract",
            str(config.ss_migration_contract),
            "--stock_slat_freeze",
            str(config.stock_slat_freeze),
            "--output_dir",
            str(paths.inference_dir),
            "--seeds",
            str(config.seed),
            "--weights",
            "ema",
            "--amp_dtype",
            config.amp_dtype,
            "--device",
            "cuda",
        ]
    return [
        (
            "raw_cache",
            raw_command,
            raw_manifest,
        ),
        (
            "runtime_o",
            runtime_command,
            runtime_manifest,
        ),
        (
            "dino_only_input",
            [
                python,
                "-u",
                "-m",
                "pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs",
                "--runtime_input_manifest",
                str(runtime_manifest),
                "--output_dir",
                str(paths.model_input_dir),
                "--device",
                "cuda",
                "--resume",
            ],
            model_manifest,
        ),
        (
            inference_name,
            inference_command,
            inference_manifest,
        ),
        (
            "world_mesh_bundle",
            [
                python,
                "-u",
                "-m",
                "pose_point_depth_mv.package_coarsemodel_real_no_vggt_results",
                "--inference_manifest",
                str(inference_manifest),
                "--runtime_input_manifest",
                str(runtime_manifest),
                "--raw_cache_report",
                str(raw_manifest),
                "--output_dir",
                str(paths.bundle_dir),
            ],
            paths.bundle_dir / "report.json",
        ),
    ]


def _load_passed_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"stage did not produce its report: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise RuntimeError(f"stage report did not pass: {path}")
    return report


def _validate_deployment_runtime_quality(
    report: dict[str, Any], *, geometry_mode: str, report_path: Path
) -> None:
    """Refuse inference when the actual selected views fail the deployment gate."""

    if geometry_mode != "pose_mask":
        return
    objects = list(report.get("objects") or [])
    if len(objects) != 1:
        raise RuntimeError(
            f"phone deployment expected one runtime object, got {len(objects)}: "
            f"{report_path}"
        )
    quality = objects[0].get("input_quality")
    if not isinstance(quality, dict):
        raise RuntimeError(
            "pose-mask runtime selected-view quality is missing; refusing to "
            f"encode posed DINO: {report_path}"
        )
    if quality.get("formal_input_passed") is not True:
        failed = [
            name for name, passed in (quality.get("checks") or {}).items() if not passed
        ]
        raise RuntimeError(
            "pose-mask runtime final selected views failed input quality: "
            f"checks={failed} values={quality.get('values')} report={report_path}"
        )


def _subprocess_environment(config: ARReconstructionConfig) -> dict[str, str]:
    environment = os.environ.copy()
    python_bin = str(Path(config.python).expanduser().resolve().parent)
    inherited_path = environment.get("PATH", "")
    environment.update(
        {
            # Calling a Conda Python by absolute path does not activate its
            # environment.  Native extensions still need tools such as ninja
            # from the matching bin directory.
            "PATH": os.pathsep.join(
                part for part in (python_bin, inherited_path) if part
            ),
            "CUDA_VISIBLE_DEVICES": str(config.gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "ATTN_BACKEND": "flash_attn",
            "SPCONV_ALGO": "native",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "NUMBA_CACHE_DIR": "/tmp/numba_cache",
            "TORCH_EXTENSIONS_DIR": "/tmp/torch_extensions",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return environment


def _run_stage(
    *,
    name: str,
    command: list[str],
    expected_report: Path,
    log_path: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        header = f"\n[{utc_now()}] stage={name} command={' '.join(command)}\n"
        log.write(header)
        log.flush()
        print(header.rstrip(), flush=True)
        process = subprocess.Popen(
            command,
            cwd="/home/zjr/Tracker",
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[ar_reconstruction:{name}] {line}", end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"reconstruction stage {name} exited with code {return_code}; log={log_path}"
        )
    return _load_passed_report(expected_report)


def run_ar_object_frame_selection(
    *,
    session_id: str,
    dataset_dir: Path,
    output_root: Path,
    config: ARReconstructionConfig,
) -> dict[str, Any]:
    """Run only the phone deployment's all-candidate -> final-view frontend.

    This deliberately reuses the first two commands of
    :func:`run_ar_object_reconstruction`: the immutable capture adapter and the
    authoritative runtime-O selector.  It stops before posed-DINO, SS, SLat and
    Mesh inference, but it still enforces the same final selected-view quality
    gate as the phone server.
    """

    config.validate()
    paths = ReconstructionPaths.build(output_root.resolve(), session_id)
    report_path = paths.run_dir / "phone_frame_selection_report.json"
    if report_path.is_file():
        existing = _load_passed_report(report_path)
        if existing.get("geometry_mode") != config.geometry_mode:
            raise RuntimeError(
                "existing frame selection uses a different geometry mode: "
                f"{report_path}"
            )
        return existing

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    environment = _subprocess_environment(config)
    combined_log = paths.run_dir / "phone_frame_selection.log"
    stage_reports: dict[str, str] = {}
    raw_report: dict[str, Any] | None = None
    runtime_report: dict[str, Any] | None = None
    for name, command, expected_report in reconstruction_commands(
        dataset_dir.resolve(), paths, config
    )[:2]:
        stage_report = _run_stage(
            name=name,
            command=command,
            expected_report=expected_report,
            log_path=combined_log,
            environment=environment,
        )
        stage_reports[name] = str(expected_report.resolve())
        if name == "raw_cache":
            raw_report = stage_report
        elif name == "runtime_o":
            if not config.diagnostic_bypass_pose_mask_quality:
                _validate_deployment_runtime_quality(
                    stage_report,
                    geometry_mode=config.geometry_mode,
                    report_path=expected_report,
                )
            runtime_report = stage_report

    if runtime_report is None:
        raise RuntimeError("runtime-O selection stage did not run")
    objects = list(runtime_report.get("objects") or [])
    if len(objects) != 1:
        raise RuntimeError(
            f"phone frame selection expected one object, got {len(objects)}"
        )
    selected = objects[0]
    capture_path = dataset_dir.resolve() / "capture_report.json"
    capture = _load_passed_report(capture_path)
    candidate_names = list(capture.get("selected_frame_names") or [])
    frontend_to_capture: dict[str, str] = {}
    if raw_report is not None:
        raw_objects = list(raw_report.get("objects") or [])
        if len(raw_objects) == 1:
            frontend_to_capture = {
                str(camera["frame_name"]): str(camera["source_frame_name"])
                for camera in raw_objects[0].get("cameras") or []
                if camera.get("frame_name") and camera.get("source_frame_name")
            }
    selected_frontend_names = list(selected.get("selected_frame_names") or [])
    selected_indices = list(selected.get("selected_source_view_indices") or [])
    selected_capture_names = [
        frontend_to_capture.get(
            name,
            candidate_names[int(index)]
            if 0 <= int(index) < len(candidate_names)
            else name,
        )
        for name, index in zip(selected_frontend_names, selected_indices)
    ]
    result = {
        "format": FRAME_SELECTION_FORMAT,
        "created_at_utc": utc_now(),
        "passed": True,
        "session_id": session_id,
        "geometry_mode": config.geometry_mode,
        "point_cloud_consumed": bool(selected.get("point_cloud_consumed")),
        "candidate_frame_count": len(candidate_names),
        "candidate_frame_names": candidate_names,
        "selected_view_count": int(selected["selected_view_count"]),
        "selected_source_view_indices": selected_indices,
        "selected_frame_names": selected_frontend_names,
        "selected_frontend_frame_names": selected_frontend_names,
        "selected_capture_frame_names": selected_capture_names,
        "frame_name_namespaces": {
            "frontend": "raw adapter names such as view_0007.png",
            "capture": "phone upload names such as frame_0007.jpg",
        },
        "view_selection": selected.get("view_selection"),
        "input_quality": selected.get("input_quality"),
        "formal_input_passed": bool(
            (selected.get("input_quality") or {}).get("formal_input_passed")
        ),
        "diagnostic_bypass_pose_mask_quality": bool(
            config.diagnostic_bypass_pose_mask_quality
        ),
        "prepared_rgb_paths": list(selected.get("prepared_rgb_paths") or []),
        "prepared_mask_paths": list(selected.get("prepared_mask_paths") or []),
        "capture_report": str(capture_path),
        "stage_reports": stage_reports,
        "log": str(combined_log.resolve()),
        "stopped_before_model_inference": True,
        "scope_guard": (
            "Exact phone deployment capture/raw/runtime-O selection path only. "
            "All captured pose+mask frames are candidates; the legacy 18-frame "
            "sample is QC-only. No DINO, SS, SLat or Mesh inference ran."
        ),
    }
    if result["candidate_frame_count"] < result["selected_view_count"]:
        raise RuntimeError("selected view count exceeds capture candidate count")
    if len(result["selected_frame_names"]) != result["selected_view_count"]:
        raise RuntimeError("runtime report selected-frame count is inconsistent")
    write_json(report_path, result)
    return result


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _obj_to_glb(source: Path, destination: Path) -> None:
    loaded = trimesh.load(source, force="scene", process=False)
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    if not scene.geometry or not any(len(mesh.vertices) for mesh in scene.geometry.values()):
        raise RuntimeError(f"cannot export an empty reconstructed mesh: {source}")
    payload = scene.export(file_type="glb")
    if not isinstance(payload, (bytes, bytearray)):
        raise RuntimeError("trimesh GLB exporter returned a non-binary payload")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_bytes(bytes(payload))
    temporary.replace(destination)


def _materialize_final_meshes(
    paths: ReconstructionPaths, bundle: dict[str, Any]
) -> dict[str, str]:
    cases = list(bundle.get("cases", []))
    if len(cases) != 1:
        raise RuntimeError(f"single-object reconstruction produced {len(cases)} bundle cases")
    case = cases[0]
    runtime_source = Path(case["predicted_runtime_o"]["path"])
    world_source = Path(case["predicted_sparse_world"]["path"])
    runtime_obj = paths.final_dir / "reconstructed_object_runtime_o.obj"
    world_obj = paths.final_dir / "reconstructed_object_world.obj"
    world_glb = paths.final_dir / "reconstructed_object_world.glb"
    mobile_overlay = paths.final_dir / "reconstructed_object_world.armesh"
    if not runtime_obj.is_file():
        _copy_atomic(runtime_source, runtime_obj)
    if not world_obj.is_file():
        _copy_atomic(world_source, world_obj)
    if not world_glb.is_file():
        _obj_to_glb(world_obj, world_glb)
    mobile_overlay_report = mobile_overlay.with_suffix(mobile_overlay.suffix + ".json")
    if not mobile_overlay.is_file() or not mobile_overlay_report.is_file():
        build_mobile_overlay_mesh(world_obj, mobile_overlay)
    return {
        "runtime_o_obj": str(runtime_obj.resolve()),
        "world_obj": str(world_obj.resolve()),
        "world_glb": str(world_glb.resolve()),
        "mobile_overlay": str(mobile_overlay.resolve()),
        "mobile_overlay_report": str(
            mobile_overlay_report.resolve()
        ),
    }


def _ensure_mobile_overlay(meshes: dict[str, Any]) -> bool:
    """Backfill the display-only phone asset for completed reconstructions."""

    world_obj = Path(str(meshes["world_obj"]))
    mobile_overlay = world_obj.with_name("reconstructed_object_world.armesh")
    report = mobile_overlay.with_suffix(mobile_overlay.suffix + ".json")
    changed = False
    if not mobile_overlay.is_file() or not report.is_file():
        build_mobile_overlay_mesh(world_obj, mobile_overlay)
        changed = True
    expected = {
        "mobile_overlay": str(mobile_overlay.resolve()),
        "mobile_overlay_report": str(report.resolve()),
    }
    for key, value in expected.items():
        if meshes.get(key) != value:
            meshes[key] = value
            changed = True
    return changed


def _preview_command(
    paths: ReconstructionPaths,
    config: ARReconstructionConfig,
    world_obj: Path,
) -> tuple[list[str], Path]:
    report_path = paths.final_dir / "previews" / "preview_report.json"
    return [
        str(config.python),
        "-u",
        "-m",
        "pose_point_depth_mv.render_ar_object_mesh_previews",
        "--mesh",
        str(world_obj),
        "--output_dir",
        str(report_path.parent),
        "--device",
        "cuda",
        "--render_frames",
        str(config.preview_render_frames),
        "--preview_count",
        str(config.preview_count),
        "--resolution",
        str(config.preview_resolution),
        "--resume",
    ], report_path


def _preview_summary(report_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    images = [str(Path(value["path"]).resolve()) for value in report.get("images", [])]
    contact_sheet = Path(str(report.get("contact_sheet", {}).get("path", "")))
    if not images or not all(Path(path).is_file() for path in images):
        raise RuntimeError(f"Mesh preview report has missing images: {report_path}")
    if not contact_sheet.is_file():
        raise RuntimeError(f"Mesh preview report has no contact sheet: {report_path}")
    return {
        "images": images,
        "contact_sheet": str(contact_sheet.resolve()),
        "report": str(report_path.resolve()),
        "render_config": report.get("render_config"),
        "display_only": True,
        "world_scale_or_pose_changed": False,
    }


def _mesh_artifacts_exist(meshes: Any) -> bool:
    required = ("runtime_o_obj", "world_obj", "world_glb")
    return isinstance(meshes, dict) and all(
        isinstance(meshes.get(name), str) and Path(meshes[name]).is_file()
        for name in required
    )


def _preview_artifacts_exist(
    previews: Any,
    config: ARReconstructionConfig,
) -> bool:
    if not isinstance(previews, dict):
        return False
    images = previews.get("images")
    render_config = previews.get("render_config")
    if (
        not isinstance(images, list)
        or len(images) != config.preview_count
        or not isinstance(render_config, dict)
        or int(render_config.get("render_frames", -1)) != config.preview_render_frames
        or int(render_config.get("preview_count", -1)) != config.preview_count
        or int(render_config.get("resolution", -1)) != config.preview_resolution
    ):
        return False
    paths = [*images, previews.get("contact_sheet"), previews.get("report")]
    return all(isinstance(path, str) and Path(path).is_file() for path in paths)


def _render_final_previews(
    *,
    paths: ReconstructionPaths,
    meshes: dict[str, str],
    config: ARReconstructionConfig,
    environment: dict[str, str],
    log_path: Path,
) -> tuple[dict[str, Any], Path]:
    command, report_path = _preview_command(
        paths,
        config,
        Path(meshes["world_obj"]),
    )
    report = _run_stage(
        name="mesh_previews",
        command=command,
        expected_report=report_path,
        log_path=log_path,
        environment=environment,
    )
    return _preview_summary(report_path, report), report_path


def run_ar_object_reconstruction(
    *,
    session_id: str,
    dataset_dir: Path,
    output_root: Path,
    config: ARReconstructionConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run or resume all deployment stages and return final world-frame meshes."""

    config.validate()
    paths = ReconstructionPaths.build(output_root.resolve(), session_id)
    final_report_path = paths.run_dir / "reconstruction_report.json"
    combined_log = paths.run_dir / "reconstruction.log"
    environment = _subprocess_environment(config)
    if final_report_path.is_file():
        existing = _load_passed_report(final_report_path)
        existing_mode = str(
            existing.get("deployment", {}).get("geometry_mode", "point_mask")
        )
        if existing_mode != config.geometry_mode:
            raise RuntimeError(
                "existing immutable reconstruction uses a different geometry mode: "
                f"existing={existing_mode} requested={config.geometry_mode} "
                f"report={final_report_path}"
            )
        existing_backend = str(
            existing.get("deployment", {}).get("slat_backend", "native_v2")
        )
        if existing_backend != config.slat_backend:
            raise RuntimeError(
                "existing immutable reconstruction uses a different SLat backend: "
                f"existing={existing_backend} requested={config.slat_backend} "
                f"report={final_report_path}"
            )
        if _mesh_artifacts_exist(existing.get("meshes")):
            meshes_changed = _ensure_mobile_overlay(existing["meshes"])
            if _preview_artifacts_exist(existing.get("previews"), config):
                if meshes_changed:
                    write_json(final_report_path, existing)
                    write_json(output_root / "latest_reconstruction.json", existing)
                return existing
            previews, preview_report_path = _render_final_previews(
                paths=paths,
                meshes=existing["meshes"],
                config=config,
                environment=environment,
                log_path=combined_log,
            )
            existing["previews"] = previews
            existing.setdefault("stage_reports", {})["mesh_previews"] = str(
                preview_report_path.resolve()
            )
            write_json(final_report_path, existing)
            write_json(output_root / "latest_reconstruction.json", existing)
            return existing

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = paths.run_dir / "progress.json"
    commands = reconstruction_commands(dataset_dir.resolve(), paths, config)
    reports: dict[str, str] = {}
    total_stages = len(commands) + 1

    def publish(payload: dict[str, Any]) -> None:
        progress = {
            "format": RECONSTRUCTION_FORMAT,
            "updated_at_utc": utc_now(),
            "session_id": session_id,
            **payload,
        }
        write_json(progress_path, progress)
        if progress_callback is not None:
            progress_callback(progress)

    publish({"status": "running", "stage": "starting", "completed_stages": 0, "total_stages": total_stages})
    try:
        for index, (name, command, expected_report) in enumerate(commands, start=1):
            publish(
                {
                    "status": "running",
                    "stage": name,
                    "completed_stages": index - 1,
                    "total_stages": total_stages,
                    "log": str(combined_log),
                }
            )
            stage_report = _run_stage(
                name=name,
                command=command,
                expected_report=expected_report,
                log_path=combined_log,
                environment=environment,
            )
            if name == "runtime_o" and not config.diagnostic_bypass_pose_mask_quality:
                _validate_deployment_runtime_quality(
                    stage_report,
                    geometry_mode=config.geometry_mode,
                    report_path=expected_report,
                )
            reports[name] = str(expected_report.resolve())
        bundle = _load_passed_report(paths.bundle_dir / "report.json")
        meshes = _materialize_final_meshes(paths, bundle)
        publish(
            {
                "status": "running",
                "stage": "mesh_previews",
                "completed_stages": len(commands),
                "total_stages": total_stages,
                "log": str(combined_log),
            }
        )
        previews, preview_report_path = _render_final_previews(
            paths=paths,
            meshes=meshes,
            config=config,
            environment=environment,
            log_path=combined_log,
        )
        reports["mesh_previews"] = str(preview_report_path.resolve())
        result = {
            "format": RECONSTRUCTION_FORMAT,
            "created_at_utc": utc_now(),
            "session_id": session_id,
            "dataset_dir": str(dataset_dir.resolve()),
            "run_dir": str(paths.run_dir.resolve()),
            "deployment": {
                **asdict(config),
                "python": str(config.python),
                "native_ss_checkpoint": str(config.native_ss_checkpoint),
                "native_slat_checkpoint": (
                    str(config.native_slat_checkpoint)
                    if config.slat_backend == "native_v2"
                    else None
                ),
                "ss_migration_contract": str(config.ss_migration_contract),
                "slat_migration_contract": (
                    str(config.slat_migration_contract)
                    if config.slat_backend == "native_v2"
                    else None
                ),
                "stock_slat_freeze": str(config.stock_slat_freeze),
            },
            "stage_reports": reports,
            "meshes": meshes,
            "previews": previews,
            "output_frame": "runtime-O and AR sparse world",
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
            "textured": False,
            "diagnostic_bypass_pose_mask_quality": bool(
                config.diagnostic_bypass_pose_mask_quality
            ),
            "formal_input_claim_allowed": not bool(
                config.diagnostic_bypass_pose_mask_quality
            ),
            "passed": True,
        }
        write_json(final_report_path, result)
        write_json(output_root / "latest_reconstruction.json", result)
        publish(
            {
                "status": "complete",
                "stage": "complete",
                "completed_stages": total_stages,
                "total_stages": total_stages,
                "reconstruction_report": str(final_report_path),
                "meshes": meshes,
                "previews": previews,
            }
        )
        return result
    except Exception as exc:
        publish(
            {
                "status": "failed",
                "stage": "failed",
                "error": repr(exc),
                "log": str(combined_log),
                "completed_stage_reports": reports,
            }
        )
        raise
