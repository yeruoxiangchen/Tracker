#!/usr/bin/env python3
"""One-command manual reconstruction with current SS30K+SLat30K and ReconViaGen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from manual_mesh_reconstruction import PACKAGE_VERSION
from manual_mesh_reconstruction.common import atomic_json, load_json, sha256_file
from manual_mesh_reconstruction.defaults import (
    ABC_R_BRIDGE,
    PRETRAINED,
    SLAT30K_CHECKPOINT,
    SLAT_STEP,
    SS30K_REPORT,
    STOCK_SLAT_FREEZE,
    validate_frozen_assets,
)
from manual_mesh_reconstruction.data_adapters.common import (
    validate_reusable_adapter_report,
)
from manual_mesh_reconstruction.pose_mask import (
    OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY,
    POSE_MASK_OBJECT_FRAME_POLICIES,
)


FORMAT = "manual_mesh_reconstruction.pipeline.v3"
ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in command)


def _run_stage(
    label: str,
    command: list[str],
    *,
    environment: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    rendered = _command_text(command)
    print(f"\n===== {label} =====\n{rendered}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] stage={label}\n{rendered}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = int(process.wait())
    if return_code != 0:
        raise RuntimeError(
            f"stage {label} exited with code {return_code}; log={log_path}"
        )


def _passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return load_json(path).get("passed") is True
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False


def _one_runtime_object(runtime_path: Path, requested: str | None) -> dict[str, Any]:
    payload = load_json(runtime_path)
    if payload.get("passed") is not True:
        raise RuntimeError(f"runtime input manifest did not pass: {runtime_path}")
    rows = list(payload.get("objects", []))
    if requested:
        rows = [row for row in rows if str(row.get("object_key")) == requested]
    if len(rows) != 1:
        raise RuntimeError(
            "manual pipeline requires exactly one object; pass --object when the "
            f"manifest contains more than one (selected={len(rows)})"
        )
    return rows[0]


def _one_inference_record(manifest_path: Path, object_key: str, seed: int) -> dict[str, Any]:
    payload = load_json(manifest_path)
    if payload.get("passed") is not True:
        raise RuntimeError(f"inference manifest did not pass: {manifest_path}")
    rows = [
        row
        for row in payload.get("objects", [])
        if str(row.get("object_key")) == object_key and int(row.get("seed", -1)) == seed
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"inference manifest has no unique object/seed row: {object_key} seed={seed}"
        )
    mesh = Path(rows[0]["mesh"]).expanduser().resolve(strict=True)
    if rows[0].get("mesh_sha256") != sha256_file(mesh):
        raise RuntimeError(f"inference Mesh hash differs: {mesh}")
    return rows[0]


def _environment(gpu: str) -> dict[str, str]:
    environment = dict(os.environ)
    additions = [str(ROOT), str(ROOT / "ReconViaGen"), str(ROOT / "ReconViaGen/wheels/vggt")]
    existing = [value for value in environment.get("PYTHONPATH", "").split(os.pathsep) if value]
    environment["PYTHONPATH"] = os.pathsep.join(additions + existing)
    if not environment.get("CUDA_HOME"):
        for candidate in (
            Path("/home/zjr/cuda-12.1"),
            Path("/usr/local/cuda"),
        ):
            if candidate.is_dir():
                environment["CUDA_HOME"] = str(candidate)
                break
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HUB_OFFLINE": environment.get("HF_HUB_OFFLINE", "1"),
            "TRANSFORMERS_OFFLINE": environment.get("TRANSFORMERS_OFFLINE", "1"),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": environment.get(
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1"
            ),
            "ATTN_BACKEND": environment.get("ATTN_BACKEND", "flash_attn"),
            "SPCONV_ALGO": environment.get("SPCONV_ALGO", "native"),
            "MPLCONFIGDIR": environment.get("MPLCONFIGDIR", "/tmp/matplotlib"),
            "NUMBA_CACHE_DIR": environment.get("NUMBA_CACHE_DIR", "/tmp/numba_cache"),
            "TORCH_EXTENSIONS_DIR": environment.get(
                "TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions"
            ),
            "PYTORCH_CUDA_ALLOC_CONF": environment.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            ),
            "OMP_NUM_THREADS": environment.get("OMP_NUM_THREADS", "2"),
            "MKL_NUM_THREADS": environment.get("MKL_NUM_THREADS", "2"),
            "OPENBLAS_NUM_THREADS": environment.get("OPENBLAS_NUM_THREADS", "1"),
        }
    )
    return environment


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--raw-cache-report",
        help="standard raw-cache report containing images, masks, K and T_W2C",
    )
    source.add_argument(
        "--runtime-input-manifest",
        help="reuse an already completed runtime-O manifest",
    )
    source.add_argument(
        "--dataset-path",
        help=(
            "phone capture/reconstruction, color+mask dataset, or Objectron "
            "dataset/clip; stage 00 constructs the raw cache automatically"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--object", default="")
    parser.add_argument("--gpu", default="0", help="one physical GPU index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selected-view-count", type=int, default=8)
    parser.add_argument(
        "--dataset-type",
        choices=("auto", "phone", "colmap", "objectron"),
        default="auto",
    )
    parser.add_argument(
        "--frame-selection",
        choices=("time_uniform", "random"),
        default="time_uniform",
    )
    parser.add_argument("--selection-seed", type=int, default=20260819)
    parser.add_argument("--session-id", default="")
    parser.add_argument(
        "--colmap-mode",
        choices=("auto", "reuse", "rebuild"),
        default="auto",
    )
    parser.add_argument("--colmap-sparse", default="")
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument(
        "--colmap-matcher",
        choices=("sequential", "exhaustive"),
        default="sequential",
    )
    parser.add_argument("--colmap-use-foreground-masks", action="store_true")
    parser.add_argument("--colmap-cpu", action="store_true")
    parser.add_argument("--objectron-clip", default="")
    parser.add_argument("--objectron-object-id", type=int, default=0)
    parser.add_argument(
        "--objectron-o",
        choices=("pose_mask", "true_object_pose"),
        default="pose_mask",
    )
    parser.add_argument(
        "--view-selection-policy",
        choices=(
            "fixed_frame_names_valid_mask",
            "lexical_even",
            "lexical_even_valid_mask_fallback",
            "time_uniform_valid_mask",
            "random_valid_mask",
            "object_azimuth_balanced_valid_mask",
            "object_spherical_farthest_valid_mask",
        ),
        default="object_spherical_farthest_valid_mask",
    )
    parser.add_argument(
        "--fixed-frame-name",
        action="append",
        default=[],
        help=(
            "exact raw-cache source frame name; repeat selected-view-count times "
            "with --view-selection-policy fixed_frame_names_valid_mask"
        ),
    )
    parser.add_argument("--gravity-up-w", type=float, nargs=3)
    parser.add_argument(
        "--pose-mask-object-frame-policy",
        choices=POSE_MASK_OBJECT_FRAME_POLICIES,
        default=OFFICIAL_COMPATIBLE_Z_UP_OBJECT_FRAME_POLICY,
        help=(
            "model-O axis contract; the default is the validated official-compatible "
            "Z-up mapping, while legacy_y_up_v1 remains available for paired replay"
        ),
    )
    parser.add_argument("--contour-width", type=int, default=3)
    parser.add_argument("--low-vram-reconviagen", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.selected_view_count) < 1:
        raise ValueError("selected-view-count must be positive")
    if int(args.contour_width) < 1:
        raise ValueError("contour-width must be positive")
    if args.view_selection_policy == "fixed_frame_names_valid_mask":
        if args.dataset_path:
            raise ValueError(
                "fixed frame replay requires --raw-cache-report so the adapter can "
                "first preserve the complete source sequence"
            )
        if len(args.fixed_frame_name) != int(args.selected_view_count):
            raise ValueError(
                "repeat --fixed-frame-name exactly selected-view-count times"
            )
    elif args.fixed_frame_name:
        raise ValueError(
            "--fixed-frame-name requires --view-selection-policy "
            "fixed_frame_names_valid_mask"
        )
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(f"output exists; pass --resume to reuse it: {output}")
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    log_path = output / "pipeline.log"
    environment = _environment(str(args.gpu))
    python = str(Path(sys.executable).resolve())
    assets = validate_frozen_assets()
    object_args = ["--object", args.object] if args.object else []

    adapter_report_path: Path | None = None
    adapter_report: dict[str, Any] | None = None
    effective_view_policy = str(args.view_selection_policy)
    if args.dataset_path:
        dataset_path = Path(args.dataset_path).expanduser().resolve(strict=True)
        adapter_dir = output / "00_dataset_adapter"
        adapter_report_path = adapter_dir / "adapter_report.json"
        adapter_command = [
            python,
            "-u",
            "-m",
            "manual_mesh_reconstruction.data_adapters.cli",
            "--dataset-path",
            str(dataset_path),
            "--dataset-type",
            str(args.dataset_type),
            "--output-dir",
            str(adapter_dir),
            "--selected-view-count",
            str(int(args.selected_view_count)),
            "--frame-selection",
            str(args.frame_selection),
            "--random-seed",
            str(int(args.selection_seed)),
            "--colmap-mode",
            str(args.colmap_mode),
            "--colmap-bin",
            str(args.colmap_bin),
            "--colmap-matcher",
            str(args.colmap_matcher),
            "--objectron-object-id",
            str(int(args.objectron_object_id)),
            "--objectron-o",
            str(args.objectron_o),
        ]
        if args.session_id:
            adapter_command.extend(["--session-id", str(args.session_id)])
        if args.colmap_sparse:
            adapter_command.extend(["--colmap-sparse", str(args.colmap_sparse)])
        if args.colmap_use_foreground_masks:
            adapter_command.append("--colmap-use-foreground-masks")
        if args.colmap_cpu:
            adapter_command.append("--colmap-cpu")
        if args.objectron_clip:
            adapter_command.extend(["--objectron-clip", str(args.objectron_clip)])
        if args.resume:
            adapter_command.append("--resume")
        if args.dry_run:
            adapter_command.append("--dry-run")
        if not _passed(adapter_report_path):
            _run_stage(
                "00 dataset adapter",
                adapter_command,
                environment=environment,
                log_path=log_path,
                dry_run=bool(args.dry_run),
            )
        # The adapter now retains every eligible RGB/mask/Pose view.  Selection
        # is deliberately deferred until runtime-O has been frozen.
        effective_view_policy = {
            "time_uniform": "time_uniform_valid_mask",
            "random": "random_valid_mask",
        }[str(args.frame_selection)]
        if args.dry_run and not adapter_report_path.is_file():
            from manual_mesh_reconstruction.data_adapters.cli import detect_type

            effective_dataset_type = (
                detect_type(dataset_path, clip_sequence=args.objectron_clip or None)
                if args.dataset_type == "auto"
                else str(args.dataset_type)
            )
            raw_path = adapter_dir / "raw_cache/raw_cache_report.json"
            runtime_path = (
                adapter_dir / "runtime_true_object_pose/runtime_input_manifest.json"
                if effective_dataset_type == "objectron"
                and args.objectron_o == "true_object_pose"
                else None
            )
        else:
            adapter_report = validate_reusable_adapter_report(
                adapter_report_path
            )
            raw_path = Path(str(adapter_report["raw_cache_report"])).resolve(strict=True)
            runtime_value = adapter_report.get("runtime_input_manifest")
            runtime_path = (
                None
                if runtime_value is None
                else Path(str(runtime_value)).resolve(strict=True)
            )
    elif args.raw_cache_report:
        raw_path = Path(args.raw_cache_report).expanduser().resolve(strict=True)
        runtime_path = None
    else:
        raw_path = None
        runtime_path = Path(args.runtime_input_manifest).expanduser().resolve(strict=True)

    if runtime_path is None:
        if raw_path is None:
            raise AssertionError("raw cache is required when runtime-O is not precomputed")
        runtime_dir = output / "01_runtime_o_pose_mask"
        runtime_path = runtime_dir / "runtime_input_manifest.json"
        command = [
            python,
            "-u",
            "-m",
            "manual_mesh_reconstruction.runtime_o",
            "--raw_cache_report",
            str(raw_path),
            "--output_dir",
            str(runtime_dir),
            "--geometry_mode",
            "pose_mask",
            "--view_selection_policy",
            effective_view_policy,
            "--selected_view_count",
            str(int(args.selected_view_count)),
            "--min_completed_objects",
            "1",
            "--pose_mask_object_frame_policy",
            str(args.pose_mask_object_frame_policy),
            "--selection_seed",
            str(int(args.selection_seed)),
            "--resume",
            *object_args,
        ]
        if args.gravity_up_w is not None:
            command.extend(["--gravity_up_w", *[str(float(v)) for v in args.gravity_up_w]])
        elif adapter_report is not None and adapter_report.get("gravity_up_W") is not None:
            command.extend(
                [
                    "--gravity_up_w",
                    *[str(float(v)) for v in adapter_report["gravity_up_W"]],
                ]
            )
        for fixed_name in args.fixed_frame_name:
            command.extend(["--fixed_frame_name", str(fixed_name)])
        if not _passed(runtime_path):
            _run_stage(
                "01 pose+mask runtime-O",
                command,
                environment=environment,
                log_path=log_path,
                dry_run=bool(args.dry_run),
            )
    if args.dry_run and not runtime_path.is_file():
        object_key = str(args.object or "<runtime object selected after stage 01>")
    else:
        runtime_row = _one_runtime_object(runtime_path, args.object or None)
        object_key = str(runtime_row["object_key"])
        if raw_path is not None and not (
            adapter_report is not None
            and adapter_report.get("geometry_mode")
            == "official_true_object_pose_oracle"
        ):
            actual_policy = runtime_row.get("pose_mask_object_frame_policy")
            if actual_policy != str(args.pose_mask_object_frame_policy):
                raise RuntimeError(
                    "runtime-O object-frame policy differs from the requested "
                    f"policy: actual={actual_policy!r} "
                    f"requested={args.pose_mask_object_frame_policy!r}; use a fresh "
                    "output directory instead of reusing a legacy runtime-O"
                )

    model_dir = output / "02_dino_only_model_input"
    model_path = model_dir / "model_input_manifest.json"
    if not _passed(model_path):
        _run_stage(
            "02 DINO-only model input",
            [
                python,
                "-u",
                "-m",
                "manual_mesh_reconstruction.model_inputs",
                "--runtime_input_manifest",
                str(runtime_path),
                "--output_dir",
                str(model_dir),
                "--pretrained",
                PRETRAINED,
                "--device",
                "cuda",
                "--resume",
                *(["--object", object_key] if not object_key.startswith("<") else []),
            ],
            environment=environment,
            log_path=log_path,
            dry_run=bool(args.dry_run),
        )

    current_dir = output / "03_current_no_vggt_ss30k_slat30k"
    current_manifest = current_dir / "inference_manifest.json"
    if not _passed(current_manifest):
        _run_stage(
            "03 current no-VGGT SS30K+SLat30K",
            [
                python,
                "-u",
                "-m",
                "manual_mesh_reconstruction.current_model",
                "--model_input_manifest",
                str(model_path),
                "--native_ss_report",
                str(SS30K_REPORT.path),
                "--native_slat_checkpoint",
                str(SLAT30K_CHECKPOINT.path),
                "--expected_slat_step",
                str(SLAT_STEP),
                "--cross_deployment_bridge_report",
                str(ABC_R_BRIDGE.path),
                "--stock_slat_freeze",
                str(STOCK_SLAT_FREEZE.path),
                "--output_dir",
                str(current_dir),
                "--pretrained",
                PRETRAINED,
                "--seeds",
                str(int(args.seed)),
                "--weights",
                "ema",
                "--device",
                "cuda",
                "--amp_dtype",
                "bf16",
                *(["--object", object_key] if not object_key.startswith("<") else []),
            ],
            environment=environment,
            log_path=log_path,
            dry_run=bool(args.dry_run),
        )

    recon_dir = output / "04_strict_reconviagen"
    recon_manifest = recon_dir / "inference_manifest.json"
    if not _passed(recon_manifest):
        command = [
            python,
            "-u",
            "-m",
            "manual_mesh_reconstruction.reconviagen",
            "--runtime_input_manifest",
            str(runtime_path),
            "--output_dir",
            str(recon_dir),
            "--pretrained",
            PRETRAINED,
            "--seeds",
            str(int(args.seed)),
            "--device",
            "cuda",
            *(["--object", object_key] if not object_key.startswith("<") else []),
        ]
        if args.low_vram_reconviagen:
            command.append("--low_vram")
        _run_stage(
            "04 strict original ReconViaGen",
            command,
            environment=environment,
            log_path=log_path,
            dry_run=bool(args.dry_run),
        )

    if args.dry_run:
        plan = {
            "format": FORMAT,
            "package_version": PACKAGE_VERSION,
            "dry_run": True,
            "output_dir": str(output),
            "runtime_input_manifest": str(runtime_path),
            "object_key": object_key,
            "pose_mask_object_frame_policy": str(
                args.pose_mask_object_frame_policy
            ),
            "o_estimation_order": (
                "all eligible input views -> freeze O -> select model views"
            ),
            "frozen_assets": assets,
        }
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return plan

    current = _one_inference_record(current_manifest, object_key, int(args.seed))
    recon = _one_inference_record(recon_manifest, object_key, int(args.seed))
    current_mesh = Path(current["mesh"]).resolve(strict=True)
    current_result = Path(current["result"]).resolve(strict=True)
    recon_mesh = Path(recon["mesh"]).resolve(strict=True)

    contours_dir = output / "05_current_original_camera_contours"
    contours_report = contours_dir / "report.json"
    if not _passed(contours_report):
        _run_stage(
            "05 current Mesh original-camera contours",
            [
                python,
                "-u",
                "-m",
                "manual_mesh_reconstruction.contours",
                "--runtime_input_manifest",
                str(runtime_path),
                "--mesh_o",
                str(current_mesh),
                "--mesh_frame_report",
                str(current_result),
                "--output_dir",
                str(contours_dir),
                "--object",
                object_key,
                "--contour_width",
                str(int(args.contour_width)),
                "--method_label",
                "no-VGGT SS30K+SLat30K",
                "--overview_name",
                "SS30K_SLat30K_原相机轮廓总览.png",
                "--resume",
            ],
            environment=environment,
            log_path=log_path,
            dry_run=False,
        )

    preview_root = output / "06_mesh_previews"
    preview_reports = {}
    for label, mesh in (("current", current_mesh), ("reconviagen", recon_mesh)):
        destination = preview_root / label
        report = destination / "preview_report.json"
        if not _passed(report):
            _run_stage(
                f"06 {label} Mesh preview",
                [
                    python,
                    "-u",
                    "-m",
                    "manual_mesh_reconstruction.render_mesh",
                    "--mesh",
                    str(mesh),
                    "--output_dir",
                    str(destination),
                    "--device",
                    "cuda",
                    "--method_label",
                    (
                        "no-VGGT SS30K+SLat30K"
                        if label == "current"
                        else "strict original ReconViaGen"
                    ),
                    "--resume",
                ],
                environment=environment,
                log_path=log_path,
                dry_run=False,
            )
        preview_reports[label] = {
            "path": str(report),
            "sha256": sha256_file(report),
        }

    report = {
        "format": FORMAT,
        "package_version": PACKAGE_VERSION,
        "created_at_utc": utc_now(),
        "passed": True,
        "object_key": object_key,
        "seed": int(args.seed),
        "input": {
            "dataset_path": (
                None
                if not args.dataset_path
                else str(Path(args.dataset_path).expanduser().resolve())
            ),
            "dataset_adapter_report": (
                None if adapter_report_path is None else str(adapter_report_path)
            ),
            "dataset_adapter_report_sha256": (
                None
                if adapter_report_path is None or not adapter_report_path.is_file()
                else sha256_file(adapter_report_path)
            ),
            "raw_cache_report": None if raw_path is None else str(raw_path),
            "runtime_input_manifest": str(runtime_path),
            "runtime_input_manifest_sha256": sha256_file(runtime_path),
            "geometry_mode": (
                str(adapter_report.get("geometry_mode"))
                if adapter_report is not None
                else ("pose_mask" if raw_path is not None else "precomputed")
            ),
            "view_selection_policy": (
                (
                    f"adapter-deferred:{args.frame_selection}->runtime:"
                    f"{effective_view_policy}"
                    if args.dataset_path
                    else str(args.view_selection_policy)
                )
                if raw_path is not None
                else "precomputed"
            ),
            "pose_mask_object_frame_policy": (
                runtime_row.get("pose_mask_object_frame_policy")
                if raw_path is not None
                else "precomputed"
            ),
            "o_frozen_before_view_selection": (
                runtime_row.get("o_frozen_before_view_selection")
                if raw_path is not None
                else None
            ),
            "all_input_view_count_for_o": (
                runtime_row.get("all_input_view_count")
                if raw_path is not None
                else None
            ),
            "selected_model_view_count": (
                runtime_row.get("selected_view_count")
                if raw_path is not None
                else None
            ),
            "fixed_frame_names": (
                list(args.fixed_frame_name) if args.fixed_frame_name else None
            ),
        },
        "frozen_assets": assets,
        "current": {
            "identity": "official no-VGGT SS30K + official no-VGGT SLat30K",
            "inference_manifest": str(current_manifest),
            "inference_manifest_sha256": sha256_file(current_manifest),
            "mesh": str(current_mesh),
            "mesh_sha256": sha256_file(current_mesh),
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
        },
        "reconviagen": {
            "identity": "strict original ReconViaGen VGGT->Stock SS->Stock SLat",
            "inference_manifest": str(recon_manifest),
            "inference_manifest_sha256": sha256_file(recon_manifest),
            "mesh": str(recon_mesh),
            "mesh_sha256": sha256_file(recon_mesh),
        },
        "contours": {
            "report": str(contours_report),
            "report_sha256": sha256_file(contours_report),
            "projection_formula": "Mesh_O -> T_O2W -> Mesh_W -> T_W2C -> K_raw+distortion",
        },
        "previews": preview_reports,
        "scientific_scope": "manual qualitative reconstruction; no held-out metric claim",
    }
    report_path = output / "run_manifest.json"
    atomic_json(report_path, report)
    print(json.dumps({"passed": True, "report": str(report_path)}, indent=2))
    return report


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
