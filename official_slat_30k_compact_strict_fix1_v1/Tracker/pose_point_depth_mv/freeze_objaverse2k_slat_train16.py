#!/usr/bin/env python3
"""Freeze a deterministic Objaverse2K SLat Train16 fitting subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat import (
    canonical_json_sha256,
    select_worker_matrix,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


SELECTION_FORMAT = "pose_point_depth_mv.objaverse2k_slat_train_selection.v1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def build_selection(
    rows: list[dict[str, Any]],
    *,
    selection_seed: int,
    object_count: int,
) -> dict[str, Any]:
    selected, start, end = select_worker_matrix(
        rows,
        seeds=[42],
        worker_index=0,
        num_workers=1,
        expected_objects=int(object_count),
        object_selection_seed=int(selection_seed),
    )
    if (start, end) != (0, int(object_count)):
        raise RuntimeError("single-worker Train subset did not cover every object")
    selected_objects = [object_uid for object_uid, _, _ in selected]
    selected_object_set = set(selected_objects)
    selected_indices = [
        index
        for index, row in enumerate(rows)
        if str(row["object_uid"]) in selected_object_set
    ]
    if not selected_indices:
        raise RuntimeError("Train subset contains no cache rows")
    subset_rows = [rows[index] for index in selected_indices]
    subset_objects = sorted({str(row["object_uid"]) for row in subset_rows})
    if subset_objects != sorted(selected_objects):
        raise RuntimeError("selected cache indices do not reproduce the frozen objects")
    if any(int(row["support_seed"]) != 42 for row in subset_rows):
        raise RuntimeError("Objaverse2K Train16 requires the frozen seed-42 train cache")
    return {
        "selection_seed": int(selection_seed),
        "object_count": len(selected_objects),
        "sample_count": len(selected_indices),
        "support_seeds": [42],
        "selected_object_uids": selected_objects,
        "representative_uids": [uid for _, uid, _ in selected],
        "selected_indices": selected_indices,
        "indices_sha256": hashlib.sha256(
            ",".join(map(str, selected_indices)).encode("utf-8")
        ).hexdigest(),
        "selection_rule": (
            "SHA256(objaverse2k_train_eval, selection_seed, object_uid, uid), "
            "first object_count; train on every cached sequence of each selected object"
        ),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--selection_seed", type=int, default=20260814)
    parser.add_argument("--object_count", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.object_count) <= 0 or int(args.selection_seed) < 0:
        raise ValueError("object_count must be positive and selection_seed non-negative")
    cache_path = Path(args.cache_manifest).expanduser().resolve()
    lifting_path = Path(args.lifting_cache_manifest).expanduser().resolve()
    cache = load_json(cache_path)
    lifting = load_json(lifting_path)
    rows = cache.get("samples")
    lifting_rows = lifting.get("samples")
    if (
        cache.get("materialized") is not True
        or int(cache.get("object_count", -1)) != 2135
        or not isinstance(rows, list)
        or not rows
    ):
        raise RuntimeError("source is not the frozen Objaverse2K train2135 SLat cache")
    if (
        lifting.get("objaverse2k_split", {}).get("name") != "train"
        or int(lifting.get("object_count", -1)) != 2135
        or not isinstance(lifting_rows, list)
    ):
        raise RuntimeError("lifting source is not the frozen Objaverse2K train split")
    lifting_uids = {str(row["uid"]) for row in lifting_rows}
    if not {str(row["uid"]) for row in rows}.issubset(lifting_uids):
        raise RuntimeError("SLat/lifting UID join is incomplete")

    selection = build_selection(
        rows,
        selection_seed=int(args.selection_seed),
        object_count=int(args.object_count),
    )
    body = {
        "format": SELECTION_FORMAT,
        "passed": True,
        "formal": False,
        "training_overlap": True,
        "source": "Objaverse2K frozen train2135 local-rebuild SLat targets",
        "cache_manifest": str(cache_path),
        "cache_manifest_sha256": sha256_file(cache_path),
        "lifting_cache_manifest": str(lifting_path),
        "lifting_cache_manifest_sha256": sha256_file(lifting_path),
        **selection,
        "scope_guard": (
            "training-overlap fitting diagnosis only; this subset cannot establish "
            "generalization or a formal scientific claim"
        ),
    }
    report = {**body, "selection_sha256": canonical_json_sha256(body)}
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = output_dir / "selection.json"
    indices_path = output_dir / "indices.txt"
    indices_text = ",".join(map(str, selection["selected_indices"])) + "\n"
    if output_dir.exists():
        if not args.resume or not report_path.is_file() or not indices_path.is_file():
            raise FileExistsError(output_dir)
        if load_json(report_path) != report or indices_path.read_text() != indices_text:
            raise RuntimeError("existing Train16 selection differs from the frozen protocol")
    else:
        output_dir.mkdir(parents=True)
        atomic_text(report_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        atomic_text(indices_path, indices_text)
    print(
        json.dumps(
            {
                "passed": True,
                "objects": selection["object_count"],
                "samples": selection["sample_count"],
                "selection_sha256": report["selection_sha256"],
                "report": str(report_path),
                "indices": str(indices_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
