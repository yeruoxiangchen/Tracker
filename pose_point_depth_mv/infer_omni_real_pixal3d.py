#!/usr/bin/env python3
"""Run official Pixal3D on the frozen largest-mask runtime-O reference view."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch


TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from pose_point_depth_mv.compare_pixal3d_singleview_smoke import (  # noqa: E402
    CUMESH_FILL_HOLES_GUARD_VERSION,
    OFFICIAL_GEOMETRY_EXPORT,
    OFFICIAL_POSTPROCESS,
    configure_local_naf,
    guarded_cumesh_fill_holes,
    load_mesh,
    model_snapshot_identity,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (  # noqa: E402
    MANIFEST_FORMAT as RUNTIME_MANIFEST_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import utc_now  # noqa: E402
from pose_point_depth_mv.mesh_benchmark_metrics import mesh_structure_metrics  # noqa: E402
from pose_point_depth_mv.omni_real_benchmark_common import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    load_json,
    object_key,
    resolve_torch_device,
    select_rows,
    sha256_file,
)


REPORT_FORMAT = "pose_point_depth_mv.omni_real_pixal3d_official_inference.v1"
MANIFEST_FORMAT = "pose_point_depth_mv.omni_real_pixal3d_inference_manifest.v1"
LEGACY_IMAGE_ONLY_RUNTIME_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_runtime_input_manifest.v1"
)
IMAGE_ONLY_RUNTIME_MANIFEST_FORMATS = {
    LEGACY_IMAGE_ONLY_RUNTIME_MANIFEST_FORMAT,
    RUNTIME_MANIFEST_FORMAT,
}
PIXAL3D_ATTENTION_ENV = {
    # The pinned Pixal3D environment intentionally uses PyTorch SDPA.  It does
    # not install flash-attn, while the surrounding ReconViaGen jobs commonly
    # export ATTN_BACKEND=flash_attn.  Never inherit that unrelated backend.
    "ATTN_BACKEND": "sdpa",
    "SPARSE_ATTN_BACKEND": "sdpa",
}


def _pixal3d_worker_environment(
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    environment.update(PIXAL3D_ATTENTION_ENV)
    return environment


def _configure_pixal3d_attention() -> None:
    os.environ.update(PIXAL3D_ATTENTION_ENV)


def parse_csv_int(value: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("seeds must be non-empty and unique")
    return values


def _reference_rgba(row: dict[str, Any], destination: Path) -> Path:
    index = int(row["reference_view_index"])
    rgb_path = Path(row["prepared_rgb_paths"][index]).resolve()
    mask_path = Path(row["prepared_mask_paths"][index]).resolve()
    with Image.open(rgb_path) as rgb_handle, Image.open(mask_path) as mask_handle:
        rgb = rgb_handle.convert("RGB")
        mask = mask_handle.convert("L")
        if rgb.size != mask.size or rgb.size != (518, 518):
            raise RuntimeError(f"invalid Pixal3D reference input: {object_key(row)}")
        rgba = rgb.convert("RGBA")
        rgba.putalpha(mask)
    alpha = np.asarray(rgba)[:, :, 3]
    if not np.any(alpha > int(0.8 * 255)) or not np.any(alpha < 255):
        raise RuntimeError(f"Pixal3D reference RGBA lacks foreground/background: {object_key(row)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        with Image.open(destination) as existing:
            if not np.array_equal(np.asarray(existing.convert("RGBA")), np.asarray(rgba)):
                raise RuntimeError(f"stale Pixal3D RGBA input: {destination}")
    else:
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        rgba.save(temporary, format="PNG")
        os.replace(temporary, destination)
    return destination


def _paths(output_dir: Path, row: dict[str, Any], seed: int) -> tuple[Path, Path, Path]:
    rgba = output_dir / "inputs" / row["category"] / f"{row['object_id']}.png"
    root = output_dir / "pixal3d" / row["category"] / row["object_id"] / f"seed_{seed}"
    return rgba, root / "mesh_official_postprocessed.glb", root / "result.json"


def _reuse(
    result_path: Path,
    mesh_path: Path,
    *,
    row: dict[str, Any],
    seed: int,
    runtime_sha256: str,
    rgba_sha256: str,
    inference_config_sha256: str,
    camera_params: dict[str, float],
) -> dict[str, Any] | None:
    if not result_path.is_file() or not mesh_path.is_file():
        return None
    result = load_json(result_path)
    expected = {
        "format": REPORT_FORMAT,
        "object_key": object_key(row),
        "seed": int(seed),
        "runtime_input_manifest_sha256": runtime_sha256,
        "input_rgba_sha256": rgba_sha256,
        "inference_config_sha256": inference_config_sha256,
        "camera_params": camera_params,
        "mesh_sha256": sha256_file(mesh_path),
        "geometry_export": OFFICIAL_GEOMETRY_EXPORT,
        "postprocess": OFFICIAL_POSTPROCESS,
    }
    mismatch = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"stale official Pixal3D result={mismatch}")
    return result


def _validate_saved_result(
    result_path: Path,
    mesh_path: Path,
    *,
    row: dict[str, Any],
    seed: int,
    runtime_path: Path,
    runtime_sha256: str,
    rgba_path: Path,
    rgba_sha256: str,
    model_path: str,
    snapshot: dict[str, Any],
    naf: dict[str, Any],
    inference_config_sha256: str,
) -> dict[str, Any] | None:
    if not result_path.is_file() and not mesh_path.is_file():
        return None
    if not result_path.is_file() or not mesh_path.is_file():
        raise RuntimeError(f"partial Pixal3D output: {result_path.parent}")
    result = load_json(result_path)
    camera_params = result.get("camera_params")
    camera_valid = (
        isinstance(camera_params, dict)
        and set(camera_params) == {"camera_angle_x", "distance", "mesh_scale"}
        and all(np.isfinite(float(value)) for value in camera_params.values())
    )
    expected = {
        "format": REPORT_FORMAT,
        "method": "pixal3d_official_single_reference_view",
        "object_key": object_key(row),
        "category": row["category"],
        "object_id": row["object_id"],
        "seed": int(seed),
        "runtime_input_manifest": str(runtime_path),
        "runtime_input_manifest_sha256": runtime_sha256,
        "reference_view_index": int(row["reference_view_index"]),
        "selection_policy": (
            "largest_undistorted_foreground_mask_then_earliest_selected_view"
        ),
        "best_of_view": False,
        "input_rgba": str(rgba_path),
        "input_rgba_sha256": rgba_sha256,
        "inference_config_sha256": inference_config_sha256,
        "mesh": str(mesh_path),
        "mesh_sha256": sha256_file(mesh_path),
        "model_path": model_path,
        "model_snapshot": snapshot,
        "naf": naf,
        "geometry_export": OFFICIAL_GEOMETRY_EXPORT,
        "postprocess": OFFICIAL_POSTPROCESS,
        "cumesh_fill_holes_guard": CUMESH_FILL_HOLES_GUARD_VERSION,
        "target_or_metric_consumed": False,
        "passed": True,
    }
    mismatch = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatch or not camera_valid:
        raise RuntimeError(
            f"stale official Pixal3D result={mismatch}, camera_valid={camera_valid}"
        )
    structure = result.get("structure")
    if not isinstance(structure, dict) or structure.get("mesh_success") is not True:
        raise RuntimeError(f"invalid official Pixal3D mesh report: {result_path}")
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime_input_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="TencentARC/Pixal3D")
    parser.add_argument("--naf_repo", type=Path, required=True)
    parser.add_argument("--naf_checkpoint", type=Path, required=True)
    parser.add_argument("--naf_source_manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--mesh_scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--max_num_tokens", type=int, default=49152)
    parser.add_argument("--sampling_steps", type=int, default=12)
    parser.add_argument("--object", action="append")
    parser.add_argument(
        "--isolate_objects",
        action="store_true",
        help="Run bounded object batches in fresh processes to release CUDA workspaces.",
    )
    parser.add_argument(
        "--isolate_batch_size",
        type=int,
        default=1,
        help="Maximum objects per isolated process and seed.",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[int], Path, str, dict[str, Path]]:
    runtime_path = Path(args.runtime_input_manifest).expanduser().resolve()
    runtime = load_json(runtime_path)
    if (
        runtime.get("format") not in IMAGE_ONLY_RUNTIME_MANIFEST_FORMATS
        or runtime.get("passed") is not True
    ):
        raise RuntimeError(f"runtime input manifest did not pass: {runtime_path}")
    rows = select_rows(runtime.get("objects", []), args.object)
    seeds = parse_csv_int(args.seeds)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_sha = sha256_file(runtime_path)
    rgba_by_key = {
        object_key(row): _reference_rgba(row, _paths(output_dir, row, seeds[0])[0])
        for row in rows
    }
    return runtime_path, runtime, rows, seeds, output_dir, runtime_sha, rgba_by_key


def _build_protocol(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    np.ndarray,
    dict[str, Any],
    str,
]:
    naf = configure_local_naf(
        args.naf_repo, args.naf_checkpoint, args.naf_source_manifest
    )
    snapshot = model_snapshot_identity(args.model_path)
    sampler_ss = {
        "steps": int(args.sampling_steps),
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "rescale_t": 5.0,
    }
    sampler_shape = {
        "steps": int(args.sampling_steps),
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
    }
    sampler_texture = {
        "steps": int(args.sampling_steps),
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }
    official_rotation = np.asarray(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    inference_config = {
        "model_path": str(args.model_path),
        "model_snapshot": snapshot,
        "naf": naf,
        "mesh_scale": float(args.mesh_scale),
        "resolution": int(args.resolution),
        "max_num_tokens": int(args.max_num_tokens),
        "sampling_steps": int(args.sampling_steps),
        "sparse_structure_sampler_params": sampler_ss,
        "shape_slat_sampler_params": sampler_shape,
        "texture_slat_sampler_params": sampler_texture,
        "geometry_export": OFFICIAL_GEOMETRY_EXPORT,
        "postprocess": OFFICIAL_POSTPROCESS,
        "official_axis_transform": official_rotation.tolist(),
        "reference_view_policy": (
            "largest_undistorted_foreground_mask_then_earliest_selected_view"
        ),
        "best_of_view": False,
    }
    inference_config_sha256 = canonical_sha256(inference_config)
    return (
        snapshot,
        naf,
        sampler_ss,
        sampler_shape,
        sampler_texture,
        official_rotation,
        inference_config,
        inference_config_sha256,
    )


def _write_manifest(
    *,
    output_dir: Path,
    runtime_path: Path,
    runtime_sha: str,
    model_path: str,
    snapshot: dict[str, Any],
    inference_config: dict[str, Any],
    inference_config_sha256: str,
    seeds: list[int],
    rows: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> None:
    manifest = {
        "format": MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "method": "pixal3d_official_single_reference_view",
        "runtime_input_manifest": str(runtime_path),
        "runtime_input_manifest_sha256": runtime_sha,
        "model_path": model_path,
        "model_snapshot": snapshot,
        "inference_config": inference_config,
        "inference_config_sha256": inference_config_sha256,
        "seeds": seeds,
        "object_count": len(rows),
        "record_count": len(reports),
        "objects": reports,
        "best_of_view": False,
        "target_or_metric_consumed": False,
        "passed": len(reports) == len(rows) * len(seeds),
    }
    manifest_path = output_dir / "inference_manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "passed": manifest["passed"],
        "object_count": len(rows),
        "record_count": len(reports),
        "manifest": str(manifest_path),
    }, indent=2))
    if not manifest["passed"]:
        raise SystemExit(2)


def _worker_command(
    args: argparse.Namespace, *, keys: list[str], seed: int
) -> list[str]:
    if not keys:
        raise ValueError("isolated worker requires at least one object")
    command = [
        sys.executable,
        "-u",
        "-m",
        "pose_point_depth_mv.infer_omni_real_pixal3d",
        "--runtime_input_manifest",
        str(args.runtime_input_manifest),
        "--output_dir",
        str(args.output_dir),
        "--model_path",
        str(args.model_path),
        "--naf_repo",
        str(args.naf_repo),
        "--naf_checkpoint",
        str(args.naf_checkpoint),
        "--naf_source_manifest",
        str(args.naf_source_manifest),
        "--device",
        str(args.device),
        "--seeds",
        str(int(seed)),
        "--mesh_scale",
        repr(float(args.mesh_scale)),
        "--resolution",
        str(int(args.resolution)),
        "--max_num_tokens",
        str(int(args.max_num_tokens)),
        "--sampling_steps",
        str(int(args.sampling_steps)),
        "--_worker",
    ]
    for key in keys:
        command.extend(("--object", key))
    if args.low_vram:
        command.append("--low_vram")
    return command


def _run_isolated(args: argparse.Namespace) -> None:
    (
        runtime_path,
        _,
        rows,
        seeds,
        output_dir,
        runtime_sha,
        rgba_by_key,
    ) = _load_inputs(args)
    (
        snapshot,
        naf,
        _,
        _,
        _,
        _,
        inference_config,
        inference_config_sha256,
    ) = _build_protocol(args)
    batch_size = int(args.isolate_batch_size)
    if batch_size <= 0:
        raise ValueError("--isolate_batch_size must be positive")
    total = len(rows) * len(seeds)
    completed_reports: dict[tuple[str, int], dict[str, Any]] = {}
    pending_by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
    for row in rows:
        key = object_key(row)
        rgba_path = rgba_by_key[key]
        rgba_sha = sha256_file(rgba_path)
        for seed in seeds:
            _, mesh_path, result_path = _paths(output_dir, row, seed)
            result = _validate_saved_result(
                result_path,
                mesh_path,
                row=row,
                seed=seed,
                runtime_path=runtime_path,
                runtime_sha256=runtime_sha,
                rgba_path=rgba_path,
                rgba_sha256=rgba_sha,
                model_path=str(args.model_path),
                snapshot=snapshot,
                naf=naf,
                inference_config_sha256=inference_config_sha256,
            )
            if result is not None:
                completed_reports[(key, seed)] = result
                print(
                    f"[real_pixal3d:isolate] reused object={key} seed={seed}",
                    flush=True,
                )
            else:
                pending_by_seed[seed].append(row)

    completed_count = len(completed_reports)
    for seed in seeds:
        pending = pending_by_seed[seed]
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            keys = [object_key(row) for row in batch]
            print(
                f"[real_pixal3d:isolate] start batch seed={seed} "
                f"objects={keys}",
                flush=True,
            )
            completed = subprocess.run(
                _worker_command(args, keys=keys, seed=seed),
                check=False,
                env=_pixal3d_worker_environment(),
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"isolated Pixal3D worker failed rc={completed.returncode} "
                    f"objects={keys} seed={seed}"
                )
            for row in batch:
                key = object_key(row)
                rgba_path = rgba_by_key[key]
                _, mesh_path, result_path = _paths(output_dir, row, seed)
                result = _validate_saved_result(
                    result_path,
                    mesh_path,
                    row=row,
                    seed=seed,
                    runtime_path=runtime_path,
                    runtime_sha256=runtime_sha,
                    rgba_path=rgba_path,
                    rgba_sha256=sha256_file(rgba_path),
                    model_path=str(args.model_path),
                    snapshot=snapshot,
                    naf=naf,
                    inference_config_sha256=inference_config_sha256,
                )
                if result is None:
                    raise RuntimeError(
                        f"isolated Pixal3D worker produced no result: {key} seed={seed}"
                    )
                completed_reports[(key, seed)] = result
                completed_count += 1
                print(
                    f"[real_pixal3d:isolate] {completed_count}/{total} completed "
                    f"object={key} seed={seed}",
                    flush=True,
                )
    reports = [
        completed_reports[(object_key(row), seed)]
        for row in rows
        for seed in seeds
    ]
    _write_manifest(
        output_dir=output_dir,
        runtime_path=runtime_path,
        runtime_sha=runtime_sha,
        model_path=str(args.model_path),
        snapshot=snapshot,
        inference_config=inference_config,
        inference_config_sha256=inference_config_sha256,
        seeds=seeds,
        rows=rows,
        reports=reports,
    )


def _run_in_process(args: argparse.Namespace) -> None:
    (
        runtime_path,
        _,
        rows,
        seeds,
        output_dir,
        runtime_sha,
        rgba_by_key,
    ) = _load_inputs(args)
    device = resolve_torch_device(args.device)
    (
        snapshot,
        naf,
        sampler_ss,
        sampler_shape,
        sampler_texture,
        official_rotation,
        inference_config,
        inference_config_sha256,
    ) = _build_protocol(args)
    if not PIXAL3D_ROOT.is_dir():
        raise FileNotFoundError(PIXAL3D_ROOT)
    sys.path.insert(0, str(PIXAL3D_ROOT))
    from inference import distance_from_fov, init_pipeline, o_voxel  # type: ignore

    pipeline = init_pipeline(
        model_path=args.model_path,
        device=str(device),
        low_vram=bool(args.low_vram),
        load_rembg=False,
    )
    reports: list[dict[str, Any]] = []
    try:
        for position, row in enumerate(rows, start=1):
            rgba_path = rgba_by_key[object_key(row)]
            rgba_sha = sha256_file(rgba_path)
            reference = int(row["reference_view_index"])
            with np.load(row["cache_npz"], allow_pickle=False) as geometry:
                intrinsic = np.asarray(geometry["K_feature"][reference], dtype=np.float64)
            with Image.open(rgba_path) as handle:
                image = handle.convert("RGBA")
                preprocessed = pipeline.preprocess_image(image)
            camera_angle_x = 2.0 * math.atan(
                float(preprocessed.width) / (2.0 * float(intrinsic[0, 0]))
            )
            distance = distance_from_fov(
                camera_angle_x,
                torch.tensor([-1.0, 0.0, 0.0]),
                torch.tensor([0.0, 511.0]),
                float(args.mesh_scale),
                512,
            )["distance_from_x"]
            camera_params = {
                "camera_angle_x": camera_angle_x,
                "distance": distance,
                "mesh_scale": float(args.mesh_scale),
            }
            for seed in seeds:
                _, mesh_path, result_path = _paths(output_dir, row, seed)
                reused = _reuse(
                    result_path,
                    mesh_path,
                    row=row,
                    seed=seed,
                    runtime_sha256=runtime_sha,
                    rgba_sha256=rgba_sha,
                    inference_config_sha256=inference_config_sha256,
                    camera_params=camera_params,
                )
                if reused is not None:
                    reports.append(reused)
                    continue
                if mesh_path.parent.exists():
                    raise RuntimeError(f"partial Pixal3D output: {mesh_path.parent}")
                torch.manual_seed(int(seed))
                mesh_list, latent = pipeline.run(
                    preprocessed,
                    camera_params=camera_params,
                    seed=int(seed),
                    sparse_structure_sampler_params=sampler_ss,
                    shape_slat_sampler_params=sampler_shape,
                    tex_slat_sampler_params=sampler_texture,
                    preprocess_image=False,
                    return_latent=True,
                    pipeline_type=f"{int(args.resolution)}_cascade",
                    max_num_tokens=int(args.max_num_tokens),
                )
                decoded = mesh_list[0]
                _, _, actual_resolution = latent
                with guarded_cumesh_fill_holes(o_voxel.postprocess.cumesh):
                    official_mesh = o_voxel.postprocess.to_glb(
                        vertices=decoded.vertices,
                        faces=decoded.faces,
                        attr_volume=decoded.attrs,
                        coords=decoded.coords,
                        attr_layout=pipeline.pbr_attr_layout,
                        grid_size=actual_resolution,
                        aabb=OFFICIAL_POSTPROCESS["aabb"],
                        decimation_target=OFFICIAL_POSTPROCESS["decimation_target"],
                        texture_size=OFFICIAL_POSTPROCESS["texture_size"],
                        remesh=OFFICIAL_POSTPROCESS["remesh"],
                        remesh_band=OFFICIAL_POSTPROCESS["remesh_band"],
                        remesh_project=OFFICIAL_POSTPROCESS["remesh_project"],
                        use_tqdm=True,
                    )
                official_mesh.apply_transform(official_rotation)
                mesh_path.parent.mkdir(parents=True, exist_ok=False)
                temporary = mesh_path.with_name(
                    f".{mesh_path.stem}.tmp-{os.getpid()}.glb"
                )
                try:
                    official_mesh.export(str(temporary), extension_webp=True)
                    if not temporary.is_file() or temporary.stat().st_size <= 0:
                        raise RuntimeError("official Pixal3D GLB export is empty")
                    os.replace(temporary, mesh_path)
                finally:
                    temporary.unlink(missing_ok=True)
                final_mesh = load_mesh(mesh_path)
                structure = mesh_structure_metrics(final_mesh)
                if not structure["mesh_success"]:
                    raise RuntimeError(f"official Pixal3D mesh is empty: {object_key(row)}")
                result = {
                    "format": REPORT_FORMAT,
                    "created_at_utc": utc_now(),
                    "method": "pixal3d_official_single_reference_view",
                    "object_key": object_key(row),
                    "category": row["category"],
                    "object_id": row["object_id"],
                    "seed": int(seed),
                    "runtime_input_manifest": str(runtime_path),
                    "runtime_input_manifest_sha256": runtime_sha,
                    "reference_view_index": reference,
                    "selection_policy": (
                        "largest_undistorted_foreground_mask_then_earliest_selected_view"
                    ),
                    "best_of_view": False,
                    "input_rgba": str(rgba_path),
                    "input_rgba_sha256": rgba_sha,
                    "inference_config_sha256": inference_config_sha256,
                    "camera_params": camera_params,
                    "mesh": str(mesh_path),
                    "mesh_sha256": sha256_file(mesh_path),
                    "structure": structure,
                    "model_path": str(args.model_path),
                    "model_snapshot": snapshot,
                    "naf": naf,
                    "sampling_steps": int(args.sampling_steps),
                    "geometry_export": OFFICIAL_GEOMETRY_EXPORT,
                    "postprocess": OFFICIAL_POSTPROCESS,
                    "cumesh_fill_holes_guard": CUMESH_FILL_HOLES_GUARD_VERSION,
                    "output_frame": "official Pixal3D axis transform; direct runtime-O diagnostic",
                    "target_or_metric_consumed": False,
                    "passed": True,
                }
                atomic_json(result_path, result)
                reports.append(result)
                print(
                    f"[real_pixal3d] {position}/{len(rows)} "
                    f"object={object_key(row)} seed={seed}",
                    flush=True,
                )
                del mesh_list, latent, decoded, official_mesh, final_mesh
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del image, preprocessed
    finally:
        del pipeline
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args._worker:
        if len(seeds) != 1 or len(reports) != len(rows):
            raise RuntimeError(
                "isolated Pixal3D worker must produce one record per object"
            )
        return
    _write_manifest(
        output_dir=output_dir,
        runtime_path=runtime_path,
        runtime_sha=runtime_sha,
        model_path=str(args.model_path),
        snapshot=snapshot,
        inference_config=inference_config,
        inference_config_sha256=inference_config_sha256,
        seeds=seeds,
        rows=rows,
        reports=reports,
    )


def main() -> None:
    # Must run before ``_run_in_process`` imports Pixal3D's attention config;
    # that module snapshots ATTN_BACKEND at import time.
    _configure_pixal3d_attention()
    args = make_parser().parse_args()
    if args.isolate_objects and not args._worker:
        _run_isolated(args)
    else:
        _run_in_process(args)


if __name__ == "__main__":
    main()
