#!/usr/bin/env python3
"""Freeze a final mixed no-VGGT deployment before consuming Holdout64."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ar_ss_flow.shared_object_preprocessing import canonical_json_sha256
from pose_point_depth_mv.evaluate_omni_real_no_vggt_final import (
    REPORT_FORMAT as BENCHMARK_FORMAT,
)
from pose_point_depth_mv.native_slat_genrecon_no_vggt import (
    NATIVE_SLAT_NO_VGGT_VERSION,
)
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    NATIVE_SS_NO_VGGT_EVAL,
    NATIVE_SS_NO_VGGT_VERSION,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.real_full_no_vggt_migration import (
    load_migration_contract,
    migration_summary,
    validate_destination_migration,
)


FORMAT = "pose_point_depth_mv.mixed_no_vggt_final_deployment.v1"
INFERENCE_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_native_no_vggt_mixed_inference_manifest.v1"
)


def _bound_file(value: str, expected_sha256: str, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != str(expected_sha256):
        raise RuntimeError(f"{label} path/hash binding failed: {path}")
    return path


def build_contract(
    *,
    benchmark: str | Path,
    ss_checkpoint: str | Path,
    slat_checkpoint: str | Path,
    ss_evidence: str | Path,
    ss_migration_contract: str | Path,
    slat_migration_contract: str | Path,
) -> dict[str, Any]:
    benchmark_path = Path(benchmark).expanduser().resolve()
    benchmark_payload = load_json(benchmark_path)
    if (
        benchmark_payload.get("format") != BENCHMARK_FORMAT
        or benchmark_payload.get("passed") is not True
        or benchmark_payload.get("formal") is not False
        or benchmark_payload.get("holdout64_consumed") is not False
        or benchmark_payload.get("no_vggt_decision", {}).get(
            "holdout_unlock_passed"
        )
        is not True
    ):
        raise RuntimeError("development Benchmark32 did not unlock final freezing")

    inference_binding = benchmark_payload.get("method_manifests", {}).get(
        "final_native_no_vggt"
    )
    if not isinstance(inference_binding, dict):
        raise RuntimeError("Benchmark32 lacks final no-VGGT inference binding")
    inference_path = _bound_file(
        str(inference_binding.get("path", "")),
        str(inference_binding.get("sha256", "")),
        label="Benchmark32 final inference",
    )
    inference = load_json(inference_path)
    if (
        inference.get("format") != INFERENCE_MANIFEST_FORMAT
        or inference.get("passed") is not True
        or inference.get("vggt_model_executed") is not False
    ):
        raise RuntimeError("final no-VGGT inference manifest did not pass")

    ss_path = Path(ss_checkpoint).expanduser().resolve()
    slat_path = Path(slat_checkpoint).expanduser().resolve()
    if (
        inference.get("native_ss_checkpoint_sha256") != sha256_file(ss_path)
        or inference.get("native_slat_checkpoint_sha256") != sha256_file(slat_path)
    ):
        raise RuntimeError("requested deployment checkpoints differ from Benchmark32")
    ss_payload = torch.load(ss_path, map_location="cpu")
    slat_payload = torch.load(slat_path, map_location="cpu")
    if (
        ss_payload.get("format") != NATIVE_SS_NO_VGGT_VERSION
        or slat_payload.get("format") != NATIVE_SLAT_NO_VGGT_VERSION
    ):
        raise RuntimeError("deployment checkpoints are not trained no-VGGT models")
    if (
        int(ss_payload.get("step", -1)) != 2000
        or int(slat_payload.get("step", -1)) != 2000
    ):
        raise RuntimeError("v1 final deployment freezes step 2000 for both stages")

    ss_contract = load_migration_contract(ss_migration_contract, stage="ss")
    slat_contract = load_migration_contract(slat_migration_contract, stage="slat")
    validate_destination_migration(ss_payload, ss_contract)
    validate_destination_migration(slat_payload, slat_contract)
    upstream = dict(slat_payload.get("model_summary", {}).get("upstream_native_ss", {}))
    if upstream.get("checkpoint_sha256") != sha256_file(ss_path):
        raise RuntimeError("final SLat upstream differs from final SS")

    evidence_path = Path(ss_evidence).expanduser().resolve()
    evidence = load_json(evidence_path)
    if (
        evidence.get("format") != NATIVE_SS_NO_VGGT_EVAL
        or evidence.get("passed") is not True
        or evidence.get("protocol", {}).get("checkpoint_sha256")
        != sha256_file(ss_path)
        or evidence.get("protocol", {}).get("weights") != "ema"
    ):
        raise RuntimeError("final SS evidence differs from the frozen SS")

    sampling_hashes = {
        canonical_json_sha256(row.get("sampling"))
        for row in inference.get("objects", [])
    }
    if len(sampling_hashes) != 1:
        raise RuntimeError("final no-VGGT inference used inconsistent SLat sampling")
    binding = {
        "benchmark32": {
            "path": str(benchmark_path),
            "sha256": sha256_file(benchmark_path),
            "decision": benchmark_payload["no_vggt_decision"],
        },
        "benchmark32_inference": {
            "path": str(inference_path),
            "sha256": sha256_file(inference_path),
        },
        "ss": {
            "checkpoint": str(ss_path),
            "checkpoint_sha256": sha256_file(ss_path),
            "checkpoint_step": 2000,
            "weights": "ema",
            "evidence": str(evidence_path),
            "evidence_sha256": sha256_file(evidence_path),
            "migration": migration_summary(ss_contract),
        },
        "slat": {
            "checkpoint": str(slat_path),
            "checkpoint_sha256": sha256_file(slat_path),
            "checkpoint_step": 2000,
            "weights": "ema",
            "migration": migration_summary(slat_contract),
            "sampling_sha256": next(iter(sampling_hashes)),
            "sampling": inference["objects"][0]["sampling"],
        },
        "input_contract": {
            "visual": "DINO-only multiview RGB",
            "pose": True,
            "runtime_o_point_cloud_lifting": True,
            "vggt_features": False,
            "vggt_depth": False,
            "vggt_model_executed": False,
        },
        "formal_holdout_policy": {
            "holdout64_consumed_before_freeze": False,
            "checkpoint_selection_after_holdout": False,
            "cfg_selection_after_holdout": False,
            "threshold_selection_after_holdout": False,
        },
    }
    return {
        "format": FORMAT,
        "binding": binding,
        "binding_sha256": canonical_json_sha256(binding),
        "training_complete": True,
        "benchmark32_passed": True,
        "holdout64_unlocked": True,
        "holdout64_consumed": False,
        "passed": True,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--ss_checkpoint", required=True)
    parser.add_argument("--slat_checkpoint", required=True)
    parser.add_argument("--ss_evidence", required=True)
    parser.add_argument("--ss_migration_contract", required=True)
    parser.add_argument("--slat_migration_contract", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    output = Path(args.output).expanduser().resolve()
    payload = build_contract(
        benchmark=args.benchmark,
        ss_checkpoint=args.ss_checkpoint,
        slat_checkpoint=args.slat_checkpoint,
        ss_evidence=args.ss_evidence,
        ss_migration_contract=args.ss_migration_contract,
        slat_migration_contract=args.slat_migration_contract,
    )
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"refusing to overwrite changed deployment: {output}")
    else:
        atomic_json(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
