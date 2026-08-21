#!/usr/bin/env python3
"""Run deterministic strict-render shards assigned to one GPU worker."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import TextIO

try:
    from .render_failure_taxonomy import (
        FAILURE_TAXONOMY_SCHEMA,
        INFRASTRUCTURE_FAILURES,
        canonical_failure_class,
    )
except ImportError:
    from render_failure_taxonomy import (
        FAILURE_TAXONOMY_SCHEMA,
        INFRASTRUCTURE_FAILURES,
        canonical_failure_class,
    )


TRACKER_ROOT = Path(__file__).resolve().parents[2]
BUILDER = Path(__file__).resolve().with_name(
    "build_objaverse_multiview_sparse_data.py"
)
SOURCE_SCHEMA = "tracker.mixed_mesh10k_sources.v1"
COMPLETE_MARKER_SCHEMA = "tracker.mixed_multiview_render_shard_complete.v1"
COMPLETE_MARKER = "_WORKER_COMPLETE.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Objaverse/Omni strict render shards assigned by "
            "shard_index mod num_workers."
        )
    )
    parser.add_argument("--source_plan", required=True)
    parser.add_argument("--render_root", required=True)
    parser.add_argument("--preview_root", required=True)
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--worker_index", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--sources", default="objaverse,omni")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--blender_path", required=True)
    parser.add_argument(
        "--xvfb_run_path",
        default=None,
        help=(
            "Optional xvfb-run executable. It is required for EEVEE when the "
            "Blender build cannot create a headless OpenGL context."
        ),
    )
    parser.add_argument(
        "--blender_engine",
        choices=("CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"),
        default="BLENDER_EEVEE",
    )
    parser.add_argument("--blender_samples", type=int, default=16)
    parser.add_argument(
        "--blender_bounds_tolerance",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--blender_cycles_device",
        choices=("CUDA", "OPTIX", "CPU"),
        default="CUDA",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--sequences_per_object", type=int, default=2)
    parser.add_argument("--candidate_views", type=int, default=24)
    parser.add_argument("--selected_views", type=int, default=8)
    parser.add_argument(
        "--trajectory_profile",
        choices=("limited_ar", "wide_ar"),
        default="limited_ar",
    )
    parser.add_argument(
        "--shard_start",
        type=int,
        default=0,
        help="Inclusive shard index lower bound.",
    )
    parser.add_argument(
        "--shard_end",
        type=int,
        default=0,
        help="Exclusive shard index upper bound; 0 disables the upper bound.",
    )
    parser.add_argument("--vis_count_per_shard", type=int, default=8)
    parser.add_argument(
        "--max_objects_per_shard",
        type=int,
        default=0,
        help="0 builds each full shard; a positive value is intended only for a separate smoke root.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_plan(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"unsupported source plan schema: {payload.get('schema')}")
    return payload


def task_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    failures = payload.get("failures")
    if not isinstance(samples, list) or not isinstance(failures, list):
        return -1
    return len(samples) + len(failures)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected_objects(
    source_manifest: Path,
    source_shard: dict,
    max_objects: int,
) -> dict[str, str]:
    expected_sha256 = str(source_shard["manifest_sha256"])
    actual_sha256 = sha256_file(source_manifest)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"frozen source shard changed: {source_manifest}; "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source_manifest}: expected UID-to-mesh object")
    values = {str(uid): str(mesh) for uid, mesh in payload.items()}
    if len(values) != int(source_shard["object_count"]):
        raise RuntimeError(
            f"source shard object count changed: {source_manifest}; "
            f"expected={source_shard['object_count']} actual={len(values)}"
        )
    if max_objects > 0:
        values = dict(list(values.items())[:max_objects])
    return values


def validate_render_inventory(
    manifest_path: Path,
    expected_objects: dict[str, str],
    sequences_per_object: int,
) -> tuple[bool, str]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"cannot load render manifest: {exc}"
    samples = payload.get("samples")
    failures = payload.get("failures")
    if not isinstance(samples, list) or not isinstance(failures, list):
        return False, "render manifest lacks samples/failures lists"
    if not samples:
        return False, "render manifest has no accepted samples"
    infrastructure_failures = [
        (row, canonical_failure_class(row))
        for row in failures
        if canonical_failure_class(row) in INFRASTRUCTURE_FAILURES
    ]
    if infrastructure_failures:
        first, first_canonical_class = infrastructure_failures[0]
        return (
            False,
            "render manifest contains infrastructure failures: "
            f"count={len(infrastructure_failures)}, "
            f"first_class={first_canonical_class}, "
            f"first_raw_class={first.get('failure_class')}, "
            f"first_uid={first.get('uid')}",
        )
    counts = {uid: 0 for uid in expected_objects}
    for row in [*samples, *failures]:
        uid = str(row.get("object_uid", row.get("uid", "")))
        if uid not in expected_objects:
            return False, f"unexpected rendered object UID: {uid}"
        actual_mesh = Path(str(row.get("source_glb", ""))).expanduser().resolve()
        expected_mesh = Path(expected_objects[uid]).expanduser().resolve()
        if actual_mesh != expected_mesh:
            return (
                False,
                f"source mesh mismatch for {uid}: {actual_mesh} != {expected_mesh}",
            )
        counts[uid] += 1
    wrong_counts = {
        uid: count
        for uid, count in counts.items()
        if count != sequences_per_object
    }
    if wrong_counts:
        return (
            False,
            f"per-object sequence counts differ; first={list(wrong_counts.items())[:5]}",
        )
    return True, "passed"


def marker_is_reusable(
    marker_path: Path,
    manifest_path: Path,
    *,
    source: str,
    shard_index: int,
    source_shard: dict,
    expected_objects: dict[str, str],
    expected_tasks: int,
    sequences_per_object: int,
) -> tuple[bool, str]:
    if not marker_path.is_file():
        return False, f"missing {COMPLETE_MARKER}"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid completion marker: {exc}"
    expected = {
        "schema": COMPLETE_MARKER_SCHEMA,
        "source": source,
        "shard_index": shard_index,
        "source_manifest_sha256": str(source_shard["manifest_sha256"]),
        "source_uid_sha256": str(source_shard["uid_sha256"]),
        "object_count": len(expected_objects),
        "task_count": expected_tasks,
        "sequences_per_object": sequences_per_object,
    }
    differing = [key for key, value in expected.items() if marker.get(key) != value]
    if differing:
        return False, f"completion marker differs: {differing}"
    if not manifest_path.is_file():
        return False, "render manifest is missing"
    actual_manifest_sha256 = sha256_file(manifest_path)
    if marker.get("render_manifest_sha256") != actual_manifest_sha256:
        return False, "render manifest hash differs from completion marker"
    return validate_render_inventory(
        manifest_path,
        expected_objects,
        sequences_per_object,
    )


def write_complete_marker(
    marker_path: Path,
    manifest_path: Path,
    *,
    source: str,
    shard_index: int,
    source_shard: dict,
    expected_objects: dict[str, str],
    expected_tasks: int,
    sequences_per_object: int,
) -> None:
    payload = {
        "schema": COMPLETE_MARKER_SCHEMA,
        "source": source,
        "shard_index": shard_index,
        "source_manifest_sha256": str(source_shard["manifest_sha256"]),
        "source_uid_sha256": str(source_shard["uid_sha256"]),
        "object_count": len(expected_objects),
        "task_count": expected_tasks,
        "sequences_per_object": sequences_per_object,
        "render_manifest": str(manifest_path.resolve()),
        "render_manifest_sha256": sha256_file(manifest_path),
        "failure_taxonomy": FAILURE_TAXONOMY_SCHEMA,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    temporary = marker_path.with_name(f".{marker_path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker_path)


def preserve_partial(path: Path) -> Path:
    destination = path.with_name(f"{path.name}.incomplete_{timestamp()}")
    if destination.exists():
        raise FileExistsError(f"cannot preserve partial output; target exists: {destination}")
    path.rename(destination)
    return destination


def run_and_tee(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=TRACKER_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return int(process.wait())


def builder_command(
    args: argparse.Namespace,
    source_manifest: Path,
    output_dir: Path,
    preview_dir: Path,
) -> list[str]:
    command = [
        str(Path(args.python).expanduser()),
        "-u",
        str(BUILDER),
        "--objaverse_manifest",
        str(source_manifest),
        "--output_dir",
        str(output_dir),
        "--code_output_dir",
        str(preview_dir),
        "--max_objects",
        str(args.max_objects_per_shard),
        "--start_index",
        "0",
        "--seed",
        str(args.seed),
        "--sequences_per_object",
        str(args.sequences_per_object),
        "--val_count",
        "0",
        "--vis_count",
        str(args.vis_count_per_shard),
        "--renderer",
        "blender",
        "--blender_path",
        str(Path(args.blender_path).expanduser().resolve()),
        "--blender_engine",
        args.blender_engine,
        "--blender_samples",
        str(args.blender_samples),
        "--blender_bounds_tolerance",
        str(args.blender_bounds_tolerance),
        "--blender_cycles_device",
        args.blender_cycles_device,
        "--blender_quiet",
        "--candidate_views",
        str(args.candidate_views),
        "--selected_views",
        str(args.selected_views),
        "--frame_selection_policy",
        (
            "object_azimuth_balanced"
            if args.trajectory_profile == "wide_ar"
            else "mask_pose_diverse"
        ),
        "--min_good_candidate_views",
        str(args.selected_views),
        "--image_size",
        "512",
        "--trajectory_mode",
        "ar_random",
        "--surface_points",
        "160000",
        "--voxel_resolution",
        "64",
        "--min_voxels",
        "1500",
        "--trajectory_resample_attempts",
        "24",
        "--camera_framing_margin_px",
        "12",
        "--min_complete_view_fraction",
        "0.45",
        "--min_usable_view_fraction",
        "0.70",
        "--max_clipped_view_fraction",
        "0.55",
        "--min_complete_in_frame_ratio",
        "0.95",
        "--min_usable_in_frame_ratio",
        "0.45",
        "--min_fg_pixels",
        "12000",
        "--min_fg_pixels_per_view",
        "512",
        "--min_fg_area_ratio",
        "0.004",
        "--max_fg_area_ratio",
        "0.88",
        "--min_bbox_margin_px",
        "12",
        "--max_border_touch_views",
        "2",
        "--max_bbox_area_ratio",
        "0.95",
        "--selection_min_fg_pixels_per_view",
        "512",
        "--selection_min_fg_area_ratio",
        "0.004",
        "--selection_min_bbox_margin_px",
        "12",
        "--selection_max_bbox_area_ratio",
        "0.95",
        "--enforce_projection_support",
        "--projection_support_frames",
        str(args.selected_views),
        "--min_projection_visible_points_ratio",
        "0.8",
        "--min_projection_support_ratio_mean",
        "0.5",
        "--max_projection_zero_support_ratio",
        "0.25",
        # The original quota is stateful and therefore shard-dependent.  Keep
        # all non-flat accepted samples here and enforce the same 0.22 cap once
        # globally, during the deterministic 10k freeze.
        "--max_low_texture_ratio",
        "1.0",
        "--low_texture_quota_warmup",
        "0",
    ]
    if args.trajectory_profile == "wide_ar":
        command.extend(
            [
                "--azimuth_span_min",
                "285.0",
                "--azimuth_span_max",
                "330.0",
                "--azimuth_jitter",
                "2.0",
                "--elevation_min",
                "-10.0",
                "--elevation_max",
                "45.0",
                "--elevation_drift",
                "20.0",
                "--elevation_jitter",
                "3.0",
                "--radius_drift",
                "0.20",
                "--roll_jitter",
                "8.0",
                "--min_selected_azimuth_coverage",
                "240.0",
                "--max_selected_azimuth_gap",
                "120.0",
            ]
        )
    if args.xvfb_run_path:
        command.extend(
            [
                "--xvfb_run_path",
                str(Path(args.xvfb_run_path).expanduser().resolve()),
            ]
        )
    return command


def worker_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("ATTN_BACKEND", "flash_attn")
    env.setdefault("SPCONV_ALGO", "native")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    env.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def emit(message: dict) -> None:
    print(json.dumps(message, ensure_ascii=False, sort_keys=True), flush=True)


def run_source(
    args: argparse.Namespace,
    plan_path: Path,
    plan: dict,
    source: str,
    env: dict[str, str],
) -> tuple[int, int]:
    completed = 0
    failed = 0
    source_plan = plan.get(source)
    if not isinstance(source_plan, dict):
        raise ValueError(f"source plan lacks {source}")
    for shard in source_plan["shards"]:
        index = int(shard["index"])
        if index < int(args.shard_start):
            continue
        if int(args.shard_end) > 0 and index >= int(args.shard_end):
            continue
        if index % args.num_workers != args.worker_index:
            continue
        source_manifest = plan_path.parent / str(shard["path"])
        source_objects = load_expected_objects(
            source_manifest,
            shard,
            args.max_objects_per_shard,
        )
        output_dir = Path(args.render_root).expanduser().resolve() / source / f"shard_{index:03d}"
        preview_dir = Path(args.preview_root).expanduser().resolve() / source / f"shard_{index:03d}"
        log_path = Path(args.log_root).expanduser().resolve() / source / f"shard_{index:03d}.log"
        expected_object_count = len(source_objects)
        expected_tasks = expected_object_count * int(args.sequences_per_object)
        manifest_path = output_dir / "manifest.json"
        marker_path = output_dir / COMPLETE_MARKER

        reusable, reuse_reason = marker_is_reusable(
            marker_path,
            manifest_path,
            source=source,
            shard_index=index,
            source_shard=shard,
            expected_objects=source_objects,
            expected_tasks=expected_tasks,
            sequences_per_object=int(args.sequences_per_object),
        )
        if reusable:
            emit(
                {
                    "stage": "reuse_complete_shard",
                    "source": source,
                    "shard": index,
                    "tasks": expected_tasks,
                }
            )
            completed += 1
            continue
        if not marker_path.exists() and manifest_path.is_file():
            actual_tasks = task_count(manifest_path)
            inventory_valid = False
            inventory_reason = "render manifest task count differs"
            if actual_tasks == expected_tasks:
                inventory_valid, inventory_reason = validate_render_inventory(
                    manifest_path,
                    source_objects,
                    int(args.sequences_per_object),
                )
            if actual_tasks == expected_tasks and inventory_valid:
                write_complete_marker(
                    marker_path,
                    manifest_path,
                    source=source,
                    shard_index=index,
                    source_shard=shard,
                    expected_objects=source_objects,
                    expected_tasks=expected_tasks,
                    sequences_per_object=int(args.sequences_per_object),
                )
                emit(
                    {
                        "stage": "recover_complete_manifest",
                        "source": source,
                        "shard": index,
                        "tasks": actual_tasks,
                        "failure_taxonomy": FAILURE_TAXONOMY_SCHEMA,
                    }
                )
                completed += 1
                continue
            emit(
                {
                    "stage": "existing_manifest_not_recoverable",
                    "source": source,
                    "shard": index,
                    "actual_tasks": actual_tasks,
                    "expected_tasks": expected_tasks,
                    "reason": inventory_reason,
                }
            )
        if output_dir.exists():
            emit(
                {
                    "stage": "reject_stale_or_partial_shard",
                    "source": source,
                    "shard": index,
                    "reason": reuse_reason,
                }
            )
        if output_dir.exists():
            preserved = preserve_partial(output_dir)
            emit(
                {
                    "stage": "preserve_partial_shard",
                    "source": source,
                    "shard": index,
                    "path": str(preserved),
                }
            )
        if log_path.exists():
            preserved_log = log_path.with_name(
                f"{log_path.name}.incomplete_{timestamp()}"
            )
            log_path.rename(preserved_log)

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        preview_dir.parent.mkdir(parents=True, exist_ok=True)
        emit(
            {
                "stage": "start_shard",
                "source": source,
                "shard": index,
                "trajectory_profile": args.trajectory_profile,
                "objects": expected_object_count,
                "expected_tasks": expected_tasks,
            }
        )
        command = builder_command(
            args,
            source_manifest,
            output_dir,
            preview_dir,
        )
        code = run_and_tee(command, log_path, env)
        actual_tasks = task_count(manifest_path) if manifest_path.is_file() else -1
        inventory_valid = False
        inventory_reason = "builder failed or task count differs"
        if code == 0 and actual_tasks == expected_tasks:
            inventory_valid, inventory_reason = validate_render_inventory(
                manifest_path,
                source_objects,
                int(args.sequences_per_object),
            )
        if code != 0 or actual_tasks != expected_tasks or not inventory_valid:
            emit(
                {
                    "stage": "failed_shard",
                    "source": source,
                    "shard": index,
                    "exit_code": code,
                    "actual_tasks": actual_tasks,
                    "expected_tasks": expected_tasks,
                    "inventory_reason": inventory_reason,
                    "log": str(log_path),
                }
            )
            failed += 1
            break
        write_complete_marker(
            marker_path,
            manifest_path,
            source=source,
            shard_index=index,
            source_shard=shard,
            expected_objects=source_objects,
            expected_tasks=expected_tasks,
            sequences_per_object=int(args.sequences_per_object),
        )
        emit(
            {
                "stage": "complete_shard",
                "source": source,
                "shard": index,
                "tasks": actual_tasks,
            }
        )
        completed += 1
    return completed, failed


def main() -> None:
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive")
    if args.worker_index < 0 or args.worker_index >= args.num_workers:
        raise ValueError("--worker_index must be in [0, num_workers)")
    if args.sequences_per_object <= 0:
        raise ValueError("--sequences_per_object must be positive")
    if args.max_objects_per_shard < 0:
        raise ValueError("--max_objects_per_shard must be nonnegative")
    if args.shard_start < 0:
        raise ValueError("--shard_start must be nonnegative")
    if args.shard_end < 0 or (
        args.shard_end > 0 and args.shard_end <= args.shard_start
    ):
        raise ValueError("--shard_end must be 0 or greater than --shard_start")
    if args.selected_views <= 0 or args.selected_views > args.candidate_views:
        raise ValueError("selected views must be positive and <= candidate views")
    if (
        not math.isfinite(args.blender_bounds_tolerance)
        or args.blender_bounds_tolerance <= 0
    ):
        raise ValueError("--blender_bounds_tolerance must be finite and positive")
    blender = Path(args.blender_path).expanduser().resolve()
    if not blender.is_file() or not os.access(blender, os.X_OK):
        raise FileNotFoundError(f"Blender is missing or not executable: {blender}")
    if args.xvfb_run_path:
        xvfb_run = Path(args.xvfb_run_path).expanduser().resolve()
        if not xvfb_run.is_file() or not os.access(xvfb_run, os.X_OK):
            raise FileNotFoundError(
                f"xvfb-run is missing or not executable: {xvfb_run}"
            )

    sources = [value.strip() for value in args.sources.split(",") if value.strip()]
    if not sources or any(source not in ("objaverse", "omni") for source in sources):
        raise ValueError("--sources must be objaverse, omni, or both")
    plan_path = Path(args.source_plan).expanduser().resolve()
    plan = load_plan(plan_path)
    env = worker_environment()

    total_completed = 0
    total_failed = 0
    for source in sources:
        completed, failed = run_source(
            args,
            plan_path,
            plan,
            source,
            env,
        )
        total_completed += completed
        total_failed += failed
        if failed:
            break
    emit(
        {
            "stage": "worker_complete",
            "worker_index": args.worker_index,
            "num_workers": args.num_workers,
            "completed_shards": total_completed,
            "failed_shards": total_failed,
            "trajectory_profile": args.trajectory_profile,
            "shard_start": int(args.shard_start),
            "shard_end": int(args.shard_end),
        }
    )
    raise SystemExit(0 if total_failed == 0 else 3)


if __name__ == "__main__":
    main()
