#!/usr/bin/env python3
"""Build Native-SLat training cache without loading or executing VGGT."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv import build_direct_slat_cache as _v2_builder
from pose_point_depth_mv import evaluate_native_ss_stock_slat_mesh as _ss_evidence
from pose_point_depth_mv.dino_only_condition import validate_dino_only_lifting_contract
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    NATIVE_SS_NO_VGGT_EVAL,
    NATIVE_SS_NO_VGGT_VERSION,
    build_native_ss_no_vggt_components,
    validate_native_ss_no_vggt_checkpoint,
)
from pose_point_depth_mv.no_vggt_ss_evidence import load_no_vggt_ss_evidence
from pose_point_depth_mv.omni_real_benchmark_common import load_json, sha256_file


PRECOMPUTED_DINO_ONLY_CONDITION_VERSION = (
    "pose_point_depth_mv.precomputed_dino_only_slat_condition.v1"
)


def _argument(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        prefix = f"{name}="
        return next((value[len(prefix) :] for value in sys.argv if value.startswith(prefix)), None)


def main() -> None:
    help_requested = "--help" in sys.argv or "-h" in sys.argv
    lifting_manifest = _argument("--lifting_manifest")
    if not lifting_manifest and not help_requested:
        raise ValueError("--lifting_manifest is required")
    condition_arch = _argument("--condition_arch")
    output_value = _argument("--output_dir")
    if not help_requested and condition_arch != "native_ss_genrecon_v2":
        raise ValueError(
            "no-VGGT cache builder requires --condition_arch native_ss_genrecon_v2"
        )
    if lifting_manifest:
        lifting = PoseLiftingCacheDataset(lifting_manifest, indices="all")
        validate_dino_only_lifting_contract(lifting)

    # The legacy builder imports this class inside main.  Replacing it with the
    # base image pipeline retains stock Flow/decoder models and DINO while
    # preventing construction of VGGT, BiRefNet, and DreamSim.
    from trellis import pipelines

    original_pipeline = pipelines.TrellisVGGTTo3DPipeline
    pipelines.TrellisVGGTTo3DPipeline = pipelines.TrellisImageTo3DPipeline
    try:
        _v2_builder.NATIVE_SS_GENRECON_VERSION = NATIVE_SS_NO_VGGT_VERSION
        _v2_builder.validate_native_ss_genrecon_checkpoint = (
            validate_native_ss_no_vggt_checkpoint
        )
        _v2_builder.build_native_ss_genrecon_components = (
            build_native_ss_no_vggt_components
        )
        _v2_builder.load_ss_evidence = load_no_vggt_ss_evidence
        _v2_builder.PRECOMPUTED_NATIVE_V2_CONDITION_VERSION = (
            PRECOMPUTED_DINO_ONLY_CONDITION_VERSION
        )
        _ss_evidence.NATIVE_SS_GENRECON_EVAL = NATIVE_SS_NO_VGGT_EVAL
        _v2_builder.main()
    finally:
        pipelines.TrellisVGGTTo3DPipeline = original_pipeline
    if help_requested:
        return

    # The reused v2 builder has one cosmetic provenance string that names its
    # historical producer. Rewrite only that metadata and refresh file hashes;
    # condition tensors and their tree hashes remain unchanged.
    output_dir = Path(str(output_value)).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = load_json(manifest_path)
    rewritten: dict[str, str] = {}
    source_text = "precomputed DINO-only lifting artifact; VGGT not executed"
    for row in manifest["samples"]:
        relative = str(row["condition_file"])
        condition_path = output_dir / relative
        if relative not in rewritten:
            payload = torch.load(condition_path, map_location="cpu")
            preprocessing = dict(payload.get("condition_preprocessing", {}))
            preprocessing["source"] = source_text
            preprocessing["vggt_model_executed"] = False
            payload["condition_preprocessing"] = preprocessing
            _v2_builder.atomic_torch_save(payload, condition_path)
            rewritten[relative] = sha256_file(condition_path)
        row["condition_file_sha256"] = rewritten[relative]
        row["condition_preprocessing"]["source"] = source_text
        row["condition_preprocessing"]["vggt_model_executed"] = False
    manifest["input_context"] = {
        "version": PRECOMPUTED_DINO_ONLY_CONDITION_VERSION,
        "source": source_text,
        "vggt_model_executed": False,
    }
    _v2_builder.atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
