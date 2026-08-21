#!/usr/bin/env python3
"""Freeze an existing DINO-only model-input manifest to an audited object subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import utc_now
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    index_objects,
    load_json,
    sha256_file,
    validate_bound_file,
)


SUBSET_FORMAT = "pose_point_depth_mv.dino_only_model_input_subset.v1"
SPLIT_FORMAT = "pose_point_depth_mv.hash_ranked_benchmark_subset.v1"
MANIFEST_FORMAT = "pose_point_depth_mv.omni_real_dino_only_model_input_manifest.v1"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_model_input_manifest", required=True)
    parser.add_argument("--ablation_split", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    source_path = Path(args.source_model_input_manifest).expanduser().resolve()
    split_path = Path(args.ablation_split).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    source = load_json(source_path)
    split = load_json(split_path)
    if (
        source.get("format") != MANIFEST_FORMAT
        or source.get("passed") is not True
        or source.get("vggt_model_loaded") is not False
        or source.get("vggt_model_executed") is not False
    ):
        raise RuntimeError(f"source DINO-only manifest did not pass: {source_path}")
    if split.get("format") != SPLIT_FORMAT:
        raise RuntimeError(f"ablation split format differs: {split_path}")
    reference_runtime = validate_bound_file(
        split["reference_runtime_manifest"],
        split["reference_runtime_manifest_sha256"],
        label="ablation reference runtime",
    )
    if (
        Path(str(source["runtime_input_manifest"])).resolve() != reference_runtime
        or str(source["runtime_input_manifest_sha256"])
        != str(split["reference_runtime_manifest_sha256"])
    ):
        raise RuntimeError("source DINO-only inputs and ablation split runtime differ")
    indexed = index_objects(source.get("objects", []), label="source DINO-only inputs")
    selected_keys = [str(value) for value in split["selected_object_keys"]]
    if len(selected_keys) != len(set(selected_keys)):
        raise RuntimeError("ablation split has duplicate object keys")
    missing = sorted(set(selected_keys).difference(indexed))
    if missing:
        raise RuntimeError(f"source DINO-only inputs lack selected objects: {missing}")
    rows = [indexed[key] for key in sorted(selected_keys)]
    for row in rows:
        validate_bound_file(
            row["model_input"],
            row["model_input_sha256"],
            label=f"DINO-only model input {row['object_key']}",
        )
        if (
            row.get("passed") is not True
            or row.get("vggt_model_loaded") is not False
            or row.get("vggt_model_executed") is not False
            or row.get("target_or_mesh_consumed") is not False
        ):
            raise RuntimeError(f"invalid selected DINO-only input: {row['object_key']}")
    manifest = {
        **source,
        "created_at_utc": utc_now(),
        "subset_contract": SUBSET_FORMAT,
        "source_model_input_manifest": str(source_path),
        "source_model_input_manifest_sha256": sha256_file(source_path),
        "ablation_split": str(split_path),
        "ablation_split_sha256": sha256_file(split_path),
        "selected_object_count": len(rows),
        "completed_object_count": len(rows),
        "reused_objects": [str(row["object_key"]) for row in rows],
        "objects": rows,
        "failures": [],
        "noise_position_contract": (
            "inference select_rows sorts this exact object set by object_key; paired "
            "pose+mask inference uses the same set, so position-derived noise is equal"
        ),
        "scope_guard": (
            "Read-only subset of existing point+mask DINO-only inputs for paired "
            "same-object-order noise replay; no feature or tensor is rewritten."
        ),
        "passed": len(rows) == len(selected_keys),
    }
    atomic_json(output_path, manifest)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "objects": len(rows),
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
