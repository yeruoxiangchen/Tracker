#!/usr/bin/env python3
"""Reconstruct one real capture with official Native-SS and trained SLat.

The output layout intentionally matches ``outputs/可视AR/reconstructions/*``:
raw cache, runtime-O, DINO-only input, inference, world bundle, and final
runtime-O/world OBJ+GLB previews.  Existing runtime-O builders and the audited
T_O2W packager remain authoritative.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

from pose_point_depth_mv.ar_object_capture import utc_now, write_json
from pose_point_depth_mv.ar_object_reconstruction import (
    ReconstructionPaths,
    _load_passed_report,
    _materialize_final_meshes,
    _render_final_previews,
    _run_stage,
    _subprocess_environment,
    _validate_deployment_runtime_quality,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


TRACKER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = TRACKER_ROOT / "pose_point_depth_mv/outputs/可视AR"
DEFAULT_PYTHON = Path("/home/zjr/anaconda3/envs/reconviagen/bin/python")
DEFAULT_SS_REPORT = Path(
    "/data/zjr/proobjaverse_official_native_ss_train2000_20260815_v1/"
    "dev64_step2000_eval16_64_seed424344_6gpu_v1/aggregate_v1/report.json"
)
DEFAULT_SLAT_CHECKPOINT = Path(
    "/data/zjr/slat_train2000_trajectory_archives/"
    "slat_train2000_trajectory_step10000_25000_strict_fix1_v1/"
    "checkpoints/step_025000.pt"
)
DEFAULT_BRIDGE_REPORT = Path(
    "/data/zjr/proobjaverse_official_slat_train2000_20260813_v1/"
    "eval_trajectory_step15000_20000_25000_seed424344_5gpu_strict_fix1_v1/"
    "step_025000/dev48_predicted/aggregate_v1/report.json"
)
DEFAULT_STOCK_FREEZE = Path(
    "/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/"
    "stock_slat_freeze_v2.json"
)
FORMAT = "pose_point_depth_mv.real_official_ss_slat_reconstruction.v1"
IDENTITY_FORMAT = "pose_point_depth_mv.real_official_ss_slat_reconstruction_identity.v1"


@dataclass(frozen=True)
class RuntimeConfig:
    python: Path
    gpu: str
    geometry_mode: str
    view_selection_policy: str
    selected_view_count: int
    min_object_points: int
    min_mask_observations: int
    min_mask_support_ratio: float
    gravity_up_w: tuple[float, float, float] | None
    native_ss_report: Path
    native_slat_checkpoint: Path
    expected_slat_step: int
    cross_deployment_bridge_report: Path
    stock_slat_freeze: Path
    seed: int
    amp_dtype: str
    preview_render_frames: int
    preview_count: int
    preview_resolution: int
    diagnostic_bypass_pose_mask_quality: bool
    model_o_axis_convention: str = "legacy_y_up"

    def validate(self) -> None:
        files = {
            "python": self.python,
            "native_ss_report": self.native_ss_report,
            "native_slat_checkpoint": self.native_slat_checkpoint,
            "cross_deployment_bridge_report": self.cross_deployment_bridge_report,
            "stock_slat_freeze": self.stock_slat_freeze,
        }
        missing = [f"{name}={path}" for name, path in files.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing real reconstruction dependency: " + "; ".join(missing))
        gpu_parts = [part.strip() for part in str(self.gpu).split(",") if part.strip()]
        if len(gpu_parts) != 1 or not gpu_parts[0].isdigit():
            raise ValueError("real reconstruction requires one physical GPU")
        if self.geometry_mode not in {"point_mask", "pose_mask"}:
            raise ValueError("geometry_mode must be point_mask or pose_mask")
        if self.view_selection_policy not in {
            "lexical_even",
            "lexical_even_valid_mask_fallback",
            "object_azimuth_balanced_valid_mask",
            "object_spherical_farthest_valid_mask",
        }:
            raise ValueError("unsupported runtime-O view selection policy")
        if self.selected_view_count < 8:
            raise ValueError("selected_view_count must be at least 8")
        if self.expected_slat_step <= 0:
            raise ValueError("expected_slat_step must be positive")
        if self.amp_dtype not in {"bf16", "fp16", "none"}:
            raise ValueError("amp_dtype must be bf16, fp16, or none")
        if self.model_o_axis_convention not in {"legacy_y_up", "official_z_up"}:
            raise ValueError(
                "model_o_axis_convention must be legacy_y_up or official_z_up"
            )


def _commands(
    dataset_dir: Path,
    paths: ReconstructionPaths,
    config: RuntimeConfig,
) -> list[tuple[str, list[str], Path]]:
    python = str(config.python)
    raw_manifest = paths.raw_dir / "raw_cache_report.json"
    runtime_manifest = paths.runtime_dir / "runtime_input_manifest.json"
    model_manifest = paths.model_input_dir / "model_input_manifest.json"
    inference_manifest = paths.inference_dir / "inference_manifest.json"
    raw = [
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
        raw.append("--allow_empty_points")
    raw.append("--resume")
    runtime = [
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
        "--model_o_axis_convention",
        config.model_o_axis_convention,
        "--min_completed_objects",
        "1",
    ]
    if config.geometry_mode == "point_mask":
        runtime.extend(
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
        runtime.extend(
            ["--gravity_up_w", *(f"{float(value):g}" for value in config.gravity_up_w)]
        )
    runtime.append("--resume")
    return [
        ("raw_cache", raw, raw_manifest),
        ("runtime_o", runtime, runtime_manifest),
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
            "official_ss_trained_slat",
            [
                python,
                "-u",
                "-m",
                "pose_point_depth_mv.infer_real_proobjaverse_official_ss_slat",
                "--model_input_manifest",
                str(model_manifest),
                "--native_ss_report",
                str(config.native_ss_report),
                "--native_slat_checkpoint",
                str(config.native_slat_checkpoint),
                "--expected_slat_step",
                str(config.expected_slat_step),
                "--cross_deployment_bridge_report",
                str(config.cross_deployment_bridge_report),
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
            ],
            inference_manifest,
        ),
        (
            "world_mesh_bundle",
            [
                python,
                "-u",
                "-m",
                "pose_point_depth_mv.package_real_proobjaverse_official_ss_slat_results",
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


def _identity(
    *, dataset_dir: Path, session_id: str, config: RuntimeConfig
) -> dict[str, Any]:
    return {
        "format": IDENTITY_FORMAT,
        "dataset_dir": str(dataset_dir),
        "session_id": session_id,
        "geometry_mode": config.geometry_mode,
        "view_selection_policy": config.view_selection_policy,
        "selected_view_count": config.selected_view_count,
        "min_object_points": config.min_object_points,
        "min_mask_observations": config.min_mask_observations,
        "min_mask_support_ratio": config.min_mask_support_ratio,
        "gravity_up_w": list(config.gravity_up_w) if config.gravity_up_w else None,
        "model_o_axis_convention": config.model_o_axis_convention,
        "native_ss_report": str(config.native_ss_report),
        "native_ss_report_sha256": sha256_file(config.native_ss_report),
        "native_slat_checkpoint": str(config.native_slat_checkpoint),
        "native_slat_checkpoint_sha256": sha256_file(config.native_slat_checkpoint),
        "expected_slat_step": config.expected_slat_step,
        "cross_deployment_bridge_report": str(config.cross_deployment_bridge_report),
        "cross_deployment_bridge_report_sha256": sha256_file(
            config.cross_deployment_bridge_report
        ),
        "stock_slat_freeze": str(config.stock_slat_freeze),
        "stock_slat_freeze_sha256": sha256_file(config.stock_slat_freeze),
        "seed": config.seed,
        "amp_dtype": config.amp_dtype,
        "preview_render_frames": config.preview_render_frames,
        "preview_count": config.preview_count,
        "preview_resolution": config.preview_resolution,
        "diagnostic_bypass_pose_mask_quality": bool(
            config.diagnostic_bypass_pose_mask_quality
        ),
    }


def run(
    *,
    dataset_dir: Path,
    session_id: str,
    output_root: Path,
    config: RuntimeConfig,
) -> dict[str, Any]:
    config.validate()
    dataset_dir = dataset_dir.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    paths = ReconstructionPaths.build(output_root, session_id)
    identity_path = paths.run_dir / "reconstruction_identity.json"
    requested_identity = _identity(
        dataset_dir=dataset_dir, session_id=session_id, config=config
    )
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != requested_identity:
            raise RuntimeError(f"existing real reconstruction identity differs: {identity_path}")
    elif paths.run_dir.exists() and any(paths.run_dir.iterdir()):
        raise RuntimeError(f"unbound nonempty reconstruction directory: {paths.run_dir}")
    else:
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, requested_identity)

    final_report_path = paths.run_dir / "reconstruction_report.json"
    if final_report_path.is_file():
        existing = _load_passed_report(final_report_path)
        if existing.get("identity") != requested_identity:
            raise RuntimeError("existing final reconstruction deployment differs")
        return existing

    environment = _subprocess_environment(config)  # type: ignore[arg-type]
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    combined_log = paths.run_dir / "reconstruction.log"
    progress_path = paths.run_dir / "progress.json"
    stage_reports: dict[str, str] = {}
    commands = _commands(dataset_dir, paths, config)
    total_stages = len(commands) + 1

    def publish(stage: str, completed: int, *, status: str = "running", **extra: Any) -> None:
        write_json(
            progress_path,
            {
                "format": FORMAT,
                "updated_at_utc": utc_now(),
                "session_id": session_id,
                "status": status,
                "stage": stage,
                "completed_stages": completed,
                "total_stages": total_stages,
                **extra,
            },
        )

    publish("starting", 0)
    try:
        for index, (name, command, expected_report) in enumerate(commands, start=1):
            publish(name, index - 1, log=str(combined_log))
            report = _run_stage(
                name=name,
                command=command,
                expected_report=expected_report,
                log_path=combined_log,
                environment=environment,
            )
            if (
                name == "runtime_o"
                and not config.diagnostic_bypass_pose_mask_quality
            ):
                _validate_deployment_runtime_quality(
                    report,
                    geometry_mode=config.geometry_mode,
                    report_path=expected_report,
                )
            stage_reports[name] = str(expected_report.resolve())

        bundle = _load_passed_report(paths.bundle_dir / "report.json")
        meshes = _materialize_final_meshes(paths, bundle)
        publish("mesh_previews", len(commands), log=str(combined_log))
        previews, preview_report = _render_final_previews(
            paths=paths,
            meshes=meshes,
            config=config,  # type: ignore[arg-type]
            environment=environment,
            log_path=combined_log,
        )
        stage_reports["mesh_previews"] = str(preview_report.resolve())
        result = {
            "format": FORMAT,
            "created_at_utc": utc_now(),
            "passed": True,
            "session_id": session_id,
            "dataset_dir": str(dataset_dir),
            "run_dir": str(paths.run_dir.resolve()),
            "identity": requested_identity,
            "deployment": {
                **asdict(config),
                "python": str(config.python),
                "native_ss_report": str(config.native_ss_report),
                "native_slat_checkpoint": str(config.native_slat_checkpoint),
                "cross_deployment_bridge_report": str(
                    config.cross_deployment_bridge_report
                ),
                "stock_slat_freeze": str(config.stock_slat_freeze),
                "gravity_up_w": (
                    list(config.gravity_up_w) if config.gravity_up_w else None
                ),
            },
            "stage_reports": stage_reports,
            "meshes": meshes,
            "previews": previews,
            "output_frame": "runtime-O and sparse/AR world",
            "runtime_o_conversion": "T_O2W from audited runtime input cache",
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
            "textured": False,
            "formal_claim_allowed": False,
            "scope_guard": (
                "Qualitative real-data reconstruction; legacy/reference meshes are "
                "visual-only and no target geometry enters model inference."
            ),
        }
        write_json(final_report_path, result)
        publish(
            "complete",
            total_stages,
            status="complete",
            reconstruction_report=str(final_report_path),
            meshes=meshes,
            previews=previews,
        )
        return result
    except Exception as error:
        publish(
            "failed",
            len(stage_reports),
            status="failed",
            error=repr(error),
            log=str(combined_log),
            completed_stage_reports=stage_reports,
        )
        raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--geometry_mode", choices=("point_mask", "pose_mask"), default="point_mask")
    parser.add_argument(
        "--view_selection_policy",
        choices=(
            "auto",
            "lexical_even",
            "lexical_even_valid_mask_fallback",
            "object_azimuth_balanced_valid_mask",
            "object_spherical_farthest_valid_mask",
        ),
        default="auto",
    )
    parser.add_argument("--selected_view_count", type=int, default=8)
    parser.add_argument("--min_object_points", type=int, default=32)
    parser.add_argument("--min_mask_observations", type=int, default=2)
    parser.add_argument("--min_mask_support_ratio", type=float, default=0.50)
    parser.add_argument("--gravity_up_w", type=float, nargs=3)
    parser.add_argument(
        "--model_o_axis_convention",
        choices=("legacy_y_up", "official_z_up"),
        default="legacy_y_up",
    )
    parser.add_argument("--native_ss_report", type=Path, default=DEFAULT_SS_REPORT)
    parser.add_argument("--native_slat_checkpoint", type=Path, default=DEFAULT_SLAT_CHECKPOINT)
    parser.add_argument("--expected_slat_step", type=int, default=25000)
    parser.add_argument(
        "--cross_deployment_bridge_report", type=Path, default=DEFAULT_BRIDGE_REPORT
    )
    parser.add_argument("--stock_slat_freeze", type=Path, default=DEFAULT_STOCK_FREEZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--preview_render_frames", type=int, default=24)
    parser.add_argument("--preview_count", type=int, default=6)
    parser.add_argument("--preview_resolution", type=int, default=512)
    parser.add_argument("--diagnostic_bypass_pose_mask_quality", action="store_true")
    return parser


def resolve_view_selection_policy(requested: str) -> str:
    """Resolve the versioned default without depending on model/GPU setup."""

    return (
        "object_spherical_farthest_valid_mask"
        if str(requested) == "auto"
        else str(requested)
    )


def main() -> None:
    args = make_parser().parse_args()
    policy = resolve_view_selection_policy(str(args.view_selection_policy))
    gravity = tuple(float(value) for value in args.gravity_up_w) if args.gravity_up_w else None
    if gravity is None and args.geometry_mode == "pose_mask":
        gravity = (0.0, 1.0, 0.0)
    config = RuntimeConfig(
        python=args.python.expanduser().resolve(),
        gpu=str(args.gpu),
        geometry_mode=str(args.geometry_mode),
        view_selection_policy=policy,
        selected_view_count=int(args.selected_view_count),
        min_object_points=int(args.min_object_points),
        min_mask_observations=int(args.min_mask_observations),
        min_mask_support_ratio=float(args.min_mask_support_ratio),
        gravity_up_w=gravity,
        model_o_axis_convention=str(args.model_o_axis_convention),
        native_ss_report=args.native_ss_report.expanduser().resolve(),
        native_slat_checkpoint=args.native_slat_checkpoint.expanduser().resolve(),
        expected_slat_step=int(args.expected_slat_step),
        cross_deployment_bridge_report=(
            args.cross_deployment_bridge_report.expanduser().resolve()
        ),
        stock_slat_freeze=args.stock_slat_freeze.expanduser().resolve(),
        seed=int(args.seed),
        amp_dtype=str(args.amp_dtype),
        preview_render_frames=int(args.preview_render_frames),
        preview_count=int(args.preview_count),
        preview_resolution=int(args.preview_resolution),
        diagnostic_bypass_pose_mask_quality=bool(
            args.diagnostic_bypass_pose_mask_quality
        ),
    )
    result = run(
        dataset_dir=args.dataset_dir,
        session_id=str(args.session_id),
        output_root=args.output_root,
        config=config,
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "run_dir": result["run_dir"],
                "runtime_o_obj": result["meshes"]["runtime_o_obj"],
                "world_obj": result["meshes"]["world_obj"],
                "world_glb": result["meshes"]["world_glb"],
                "preview": result["previews"]["contact_sheet"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
