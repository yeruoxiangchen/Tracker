#!/usr/bin/env python3
"""Process completed Objaverse render shards while rendering continues.

The CPU stage performs A/B/C/R post-audit, point-prior construction, and
PointPose construction independently for every render shard with a valid
completion marker.  The DINO stage watches those CPU-ready shards and invokes
the direct DINO-only lifting-cache builder on one selected GPU.

No active render directory without ``_WORKER_COMPLETE.json`` is ever read.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


RENDER_MARKER_FORMAT = "tracker.mixed_multiview_render_shard_complete.v1"
CPU_MARKER_FORMAT = "pose_point_depth_mv.objaverse5k_cpu_shard_ready.v1"
CPU_MARKER = "_CPU_SHARD_READY.json"
DINO_MARKER = "_DINO_ONLY_LIFTING_COMPLETE.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_shards(spec: str) -> set[int] | None:
    text = str(spec).strip().lower()
    if text in {"", "all"}:
        return None
    output: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            output.update(range(int(start), int(end) + 1))
        else:
            output.add(int(item))
    if not output or min(output) < 0:
        raise ValueError("--shards must select nonnegative indices")
    return output


def validate_render_marker(marker_path: Path) -> dict[str, Any]:
    marker = load_json(marker_path)
    try:
        shard_index = int(marker_path.parent.name.removeprefix("shard_"))
    except ValueError as error:
        raise ValueError(f"invalid render shard directory: {marker_path}") from error
    expected = {
        "schema": RENDER_MARKER_FORMAT,
        "source": "objaverse",
        "shard_index": shard_index,
    }
    changed = {
        key: (marker.get(key), value)
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if changed:
        raise RuntimeError(f"render completion marker differs: {changed}")
    manifest = marker_path.parent / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if sha256_file(manifest) != str(marker.get("render_manifest_sha256", "")):
        raise RuntimeError(f"render manifest hash differs from marker: {manifest}")
    marker_manifest = Path(str(marker.get("render_manifest", ""))).expanduser()
    if marker_manifest.resolve() != manifest.resolve():
        raise RuntimeError(f"render marker path differs: {marker_manifest} != {manifest}")
    return {
        "shard_index": shard_index,
        "marker": marker,
        "marker_path": marker_path.resolve(),
        "marker_sha256": sha256_file(marker_path),
        "manifest_path": manifest.resolve(),
        "manifest_sha256": sha256_file(manifest),
    }


def discover_render_shards(render_root: Path, selected: set[int] | None) -> list[dict[str, Any]]:
    rows = []
    for marker in sorted(
        (render_root / "objaverse").glob(f"shard_*/_WORKER_COMPLETE.json")
    ):
        row = validate_render_marker(marker)
        if selected is None or row["shard_index"] in selected:
            rows.append(row)
    return rows


def discover_render_shards_from_roots(
    render_roots: list[Path],
    selected: set[int] | None,
) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for root in render_roots:
        for row in discover_render_shards(root, selected):
            index = int(row["shard_index"])
            if index in by_index:
                previous = by_index[index]["manifest_path"]
                raise RuntimeError(
                    f"duplicate completed render shard={index:03d}: "
                    f"{previous} and {row['manifest_path']}"
                )
            by_index[index] = row
    return [by_index[index] for index in sorted(by_index)]


def run_command(command: list[str]) -> None:
    print("[shard_processor] exec:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def reusable_json_pair(
    first: Path,
    second: Path,
    *,
    source_manifest: Path,
) -> bool:
    if not first.is_file() or not second.is_file():
        return False
    payload = load_json(first)
    bound = Path(str(payload.get("source_manifest", ""))).expanduser()
    return bound.resolve() == source_manifest.resolve()


def locate_ab_manifest(post_dir: Path, render_manifest: Path) -> Path:
    report_path = post_dir / "report.json"
    report = load_json(report_path)
    if report.get("passed") is not True:
        raise RuntimeError(f"post-audit did not pass: {report_path}")
    candidates = []
    for row in report.get("filtered_manifests", []):
        if sorted(row.get("tiers", [])) != ["A", "B"]:
            continue
        path = Path(str(row["path"])).expanduser().resolve()
        payload = load_json(path)
        source = Path(
            str(payload.get("single_object_restructure", {}).get("source_manifest", ""))
        ).expanduser()
        if source.resolve() == render_manifest.resolve():
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one A/B manifest for {render_manifest}, got {candidates}"
        )
    payload = load_json(candidates[0])
    if not payload.get("samples"):
        raise RuntimeError(f"A/B manifest has no samples: {candidates[0]}")
    return candidates[0]


def cpu_marker_reusable(marker_path: Path, render: dict[str, Any]) -> bool:
    if not marker_path.is_file():
        return False
    marker = load_json(marker_path)
    if (
        marker.get("format") != CPU_MARKER_FORMAT
        or marker.get("render_completion_marker_sha256")
        != render["marker_sha256"]
        or marker.get("render_manifest_sha256") != render["manifest_sha256"]
        or marker.get("passed") is not True
    ):
        raise RuntimeError(f"stale CPU shard marker: {marker_path}")
    for key in ("ab_manifest", "prior_manifest", "pointpose_manifest"):
        path = Path(str(marker.get(key, "")))
        if not path.is_file() or sha256_file(path) != marker.get(f"{key}_sha256"):
            raise RuntimeError(f"CPU shard marker binding changed: {key}={path}")
    return True


def process_cpu_shard(args: argparse.Namespace, render: dict[str, Any]) -> bool:
    index = int(render["shard_index"])
    shard_root = Path(args.work_root).expanduser().resolve() / "shards" / f"shard_{index:03d}"
    marker_path = shard_root / CPU_MARKER
    if cpu_marker_reusable(marker_path, render):
        print(f"[shard_processor] CPU reuse shard={index:03d}", flush=True)
        return False

    post_dir = shard_root / "post"
    post_command = [
        str(args.python),
        "-u",
        "-m",
        "pose_point_depth_mv.dataset_tools.restructure_single_object_dataset",
        "--manifest",
        str(render["manifest_path"]),
        "--output_dir",
        str(post_dir),
        "--workers",
        str(args.audit_workers),
    ]
    for value in args.exclude_manifest:
        post_command.extend(("--exclude_manifest", str(Path(value).expanduser().resolve())))
    run_command(post_command)
    ab_manifest = locate_ab_manifest(post_dir, render["manifest_path"])

    prior_dir = shard_root / "point_prior"
    prior_manifest = prior_dir / "manifest.json"
    if reusable_json_pair(
        prior_manifest,
        prior_dir / "build_report.json",
        source_manifest=ab_manifest,
    ):
        print(f"[shard_processor] prior reuse shard={index:03d}", flush=True)
    elif prior_dir.exists():
        raise RuntimeError(f"partial/stale point-prior output: {prior_dir}")
    else:
        run_command(
            [
                str(args.python),
                "-u",
                "trellis_point_prior_mv/build_point_prior_dataset.py",
                "--source_manifest",
                str(ab_manifest),
                "--output_dir",
                str(prior_dir),
                "--indices",
                "all",
                "--max_frames",
                "8",
                "--seed",
                str(args.prior_seed),
                "--grid_transform",
                "pixal3d_rotation",
                "--num_prior_views_choices",
                "2,4,8",
                "--point_count_choices",
                "50,100,300,800,1500",
                "--min_support",
                "1",
                "--min_support_ratio",
                "0.45",
                "--dropout_min",
                "0.0",
                "--dropout_max",
                "0.65",
                "--coord_jitter",
                "1",
                "--outlier_ratio",
                "0.03",
                "--front_depth_epsilon",
                "0.02",
                "--log_every",
                str(args.log_every),
            ]
        )

    pointpose_dir = shard_root / "pointpose"
    pointpose_manifest = pointpose_dir / "manifest.json"
    if reusable_json_pair(
        pointpose_manifest,
        pointpose_dir / "cache_audit.json",
        source_manifest=ab_manifest,
    ):
        print(f"[shard_processor] PointPose reuse shard={index:03d}", flush=True)
    elif pointpose_dir.exists():
        raise RuntimeError(f"partial/stale PointPose output: {pointpose_dir}")
    else:
        run_command(
            [
                str(args.python),
                "-u",
                "reconvggt_ar_adapter_a/build_pointpose_ss_cache.py",
                "--source_manifest",
                str(ab_manifest),
                "--prior_manifest",
                str(prior_manifest),
                "--output_dir",
                str(pointpose_dir),
                "--indices",
                "all",
                "--log_every",
                str(args.log_every),
            ]
        )

    for path in (ab_manifest, prior_manifest, pointpose_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    pointpose = load_json(pointpose_manifest)
    audit = load_json(pointpose_dir / "cache_audit.json")
    if audit.get("hard_failures") != 0 or not pointpose.get("samples"):
        raise RuntimeError(f"PointPose shard failed hard gate: {pointpose_dir}")
    marker = {
        "format": CPU_MARKER_FORMAT,
        "created_at_utc": utc_now(),
        "shard_index": index,
        "render_completion_marker": str(render["marker_path"]),
        "render_completion_marker_sha256": render["marker_sha256"],
        "render_manifest": str(render["manifest_path"]),
        "render_manifest_sha256": render["manifest_sha256"],
        "ab_manifest": str(ab_manifest),
        "ab_manifest_sha256": sha256_file(ab_manifest),
        "prior_manifest": str(prior_manifest),
        "prior_manifest_sha256": sha256_file(prior_manifest),
        "pointpose_manifest": str(pointpose_manifest),
        "pointpose_manifest_sha256": sha256_file(pointpose_manifest),
        "sample_count": len(pointpose["samples"]),
        "object_count": len(
            {str(row.get("object_uid", row["uid"])) for row in pointpose["samples"]}
        ),
        "draft_ab_semantic_filter": True,
        "passed": True,
    }
    atomic_json(marker_path, marker)
    print(
        f"[shard_processor] CPU complete shard={index:03d} "
        f"samples={marker['sample_count']} objects={marker['object_count']}",
        flush=True,
    )
    return True


def discover_cpu_ready(work_root: Path, selected: set[int] | None) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((work_root / "shards").glob(f"shard_*/{CPU_MARKER}")):
        marker = load_json(path)
        if marker.get("format") != CPU_MARKER_FORMAT or marker.get("passed") is not True:
            raise RuntimeError(f"invalid CPU-ready marker: {path}")
        index = int(marker["shard_index"])
        if selected is not None and index not in selected:
            continue
        pointpose = Path(str(marker["pointpose_manifest"])).resolve()
        if not pointpose.is_file() or sha256_file(pointpose) != marker.get(
            "pointpose_manifest_sha256"
        ):
            raise RuntimeError(f"CPU-ready PointPose binding changed: {path}")
        rows.append({"shard_index": index, "marker_path": path, "marker": marker})
    return rows


def dino_marker_reusable(output_dir: Path, cpu: dict[str, Any]) -> bool:
    marker_path = output_dir / DINO_MARKER
    if not marker_path.is_file():
        return False
    marker = load_json(marker_path)
    pointpose_sha = cpu["marker"]["pointpose_manifest_sha256"]
    if (
        marker.get("source_cache_manifest_sha256") != pointpose_sha
        or marker.get("vggt_model_loaded") is not False
        or marker.get("vggt_model_executed") is not False
    ):
        raise RuntimeError(f"stale/failed DINO shard marker: {marker_path}")
    if marker.get("passed") is not True:
        # A strict build with --allow_failures disabled normally exits before a
        # marker is written.  This branch also permits deliberate failed-cache
        # audits to be retried with the unchanged run binding.
        return False
    manifest = Path(str(marker.get("manifest", "")))
    if not manifest.is_file() or sha256_file(manifest) != marker.get("manifest_sha256"):
        raise RuntimeError(f"DINO shard manifest binding changed: {marker_path}")
    return True


def process_dino_shard(args: argparse.Namespace, cpu: dict[str, Any]) -> bool:
    index = int(cpu["shard_index"])
    shard_root = Path(args.work_root).expanduser().resolve() / "shards" / f"shard_{index:03d}"
    output_dir = shard_root / "dino_only"
    if dino_marker_reusable(output_dir, cpu):
        print(f"[shard_processor] DINO reuse shard={index:03d}", flush=True)
        return False
    command = [
        str(args.python),
        "-u",
        "-m",
        "pose_point_depth_mv.dataset_tools.build_dino_only_lifting_cache_direct",
        "--source_cache_manifest",
        str(cpu["marker"]["pointpose_manifest"]),
        "--output_dir",
        str(output_dir),
        "--dino_model",
        str(args.dino_model),
        "--indices",
        "all",
        "--device",
        str(args.device),
        "--image_resolution",
        "518",
        "--foreground_margin",
        "1.10",
        "--alpha_threshold",
        "0.80",
        "--ss_context_tokens",
        str(args.ss_context_tokens),
        "--save_correct_geometry",
        "--resume",
        "--log_every",
        str(args.log_every),
    ]
    run_command(command)
    if not dino_marker_reusable(output_dir, cpu):
        raise RuntimeError(f"DINO shard did not produce completion marker: {output_dir}")
    return True


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "dino"), required=True)
    parser.add_argument("--work_root", required=True)
    parser.add_argument(
        "--render_root",
        action="append",
        default=[],
        help="Render root for CPU discovery; repeat for disjoint shard roots.",
    )
    parser.add_argument("--exclude_manifest", action="append", default=[])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--shards", default="all")
    parser.add_argument("--expected_shards", type=int, default=16)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--audit_workers", type=int, default=8)
    parser.add_argument("--prior_seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--dino_model", default="dinov2_vitl14_reg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ss_context_tokens", type=int, default=4096)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.stage == "cpu" and not args.render_root:
        raise ValueError("at least one --render_root is required for --stage cpu")
    if int(args.expected_shards) <= 0 or float(args.poll_seconds) <= 0:
        raise ValueError("expected_shards and poll_seconds must be positive")
    selected = parse_shards(args.shards)
    work_root = Path(args.work_root).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    while True:
        if args.stage == "cpu":
            available = discover_render_shards_from_roots(
                [Path(value).expanduser().resolve() for value in args.render_root],
                selected,
            )
            for row in available:
                process_cpu_shard(args, row)
            ready = discover_cpu_ready(work_root, selected)
            complete_count = len(ready)
            available_count = len(available)
        else:
            available = discover_cpu_ready(work_root, selected)
            for row in available:
                process_dino_shard(args, row)
            complete_count = sum(
                int(
                    dino_marker_reusable(
                        work_root
                        / "shards"
                        / f"shard_{int(row['shard_index']):03d}"
                        / "dino_only",
                        row,
                    )
                )
                for row in available
            )
            available_count = len(available)
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "available_shards": available_count,
                    "complete_shards": complete_count,
                    "expected_shards": int(args.expected_shards),
                    "watch": bool(args.watch),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if not args.watch or complete_count >= int(args.expected_shards):
            break
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    main()
