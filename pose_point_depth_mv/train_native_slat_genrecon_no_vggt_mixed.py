#!/usr/bin/env python3
"""Train mixed-domain no-VGGT SLat from a contracted real-Full EMA parent."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch

from pose_point_depth_mv import evaluate_native_ss_stock_slat_mesh as _ss_evidence
from pose_point_depth_mv import train_native_slat_genrecon as _v2_train
from pose_point_depth_mv.mixed_no_vggt_data import (
    DomainBalancedDistributedSampler,
    MixedNativeConditionSLatDataset,
)
from pose_point_depth_mv.native_slat_genrecon_no_vggt import (
    NATIVE_SLAT_NO_VGGT_VERSION,
    build_native_slat_no_vggt_components,
    validate_native_slat_no_vggt_checkpoint,
)
from pose_point_depth_mv.native_slat_genrecon_v2 import (
    NATIVE_SLAT_GENRECON_V2_VERSION,
)
from pose_point_depth_mv.native_ss_genrecon_no_vggt import NATIVE_SS_NO_VGGT_EVAL
from pose_point_depth_mv.no_vggt_ss_evidence import load_no_vggt_ss_evidence
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file
from pose_point_depth_mv.real_full_no_vggt_migration import (
    load_migration_contract,
    migration_summary,
    validate_destination_migration,
    validate_parent_payload,
)


def _argument(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        prefix = f"{name}="
        return next((value[len(prefix) :] for value in sys.argv if value.startswith(prefix)), None)


def _pop_argument(name: str) -> str:
    prefix = f"{name}="
    for index, value in enumerate(list(sys.argv)):
        if value.startswith(prefix):
            del sys.argv[index]
            return value[len(prefix) :]
        if value == name:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{name} requires a value")
            result = sys.argv[index + 1]
            del sys.argv[index : index + 2]
            return result
    raise ValueError(f"{name} is required")


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print("mixed-only argument: --migration_contract PATH", file=sys.stderr)
        _v2_train.main()
        return
    architecture = _argument("--architecture")
    if architecture is None:
        sys.argv.extend(("--architecture", "v2"))
    elif architecture != "v2":
        raise ValueError("mixed no-VGGT SLat freezes the unchanged v2 architecture")
    contract = load_migration_contract(
        _pop_argument("--migration_contract"), stage="slat"
    )
    init_path = _argument("--init_checkpoint")
    resume_path = _argument("--resume")
    init_weights = _argument("--init_weights") or "ema"
    if bool(init_path) == bool(resume_path):
        raise ValueError("mixed migration requires exactly one of --init_checkpoint/--resume")
    resume_micro_step = 0
    if init_path:
        resolved = Path(init_path).expanduser().resolve()
        expected = Path(str(contract["parent"]["checkpoint"])).resolve()
        if resolved != expected or sha256_file(resolved) != contract["parent"]["checkpoint_sha256"]:
            raise RuntimeError("--init_checkpoint is not the contracted real-Full parent")
        if init_weights != "ema":
            raise ValueError("mixed no-VGGT migration requires --init_weights ema")
    else:
        resume = torch.load(Path(str(resume_path)).expanduser().resolve(), map_location="cpu")
        resume_micro_step = int(resume.get("micro_step", 0))
        validate_destination_migration(resume, contract)

    def strict_checkpoint_validator(
        checkpoint: dict[str, Any],
        *,
        pretrained: str,
        stock_slat_freeze: dict[str, Any],
        upstream_native_ss: dict[str, Any],
        **_: Any,
    ) -> None:
        if checkpoint.get("format") == NATIVE_SLAT_GENRECON_V2_VERSION:
            validate_native_slat_no_vggt_checkpoint(
                checkpoint,
                pretrained=pretrained,
                stock_slat_freeze=stock_slat_freeze,
                upstream_native_ss=upstream_native_ss,
                allow_v2_parent=True,
            )
            validate_parent_payload(checkpoint, contract, stage="slat")
        else:
            validate_native_slat_no_vggt_checkpoint(
                checkpoint,
                pretrained=pretrained,
                stock_slat_freeze=stock_slat_freeze,
                upstream_native_ss=upstream_native_ss,
                allow_v2_parent=False,
            )
            validate_destination_migration(checkpoint, contract)

    def strict_component_builder(**kwargs: Any):
        sampler, model, decoder, summary, defaults, normalization = (
            build_native_slat_no_vggt_components(**kwargs)
        )
        summary = {**summary, "migration_contract": migration_summary(contract)}
        return sampler, model, decoder, summary, defaults, normalization

    def balanced_sampler_factory(rows, *, num_replicas, rank, seed):
        return DomainBalancedDistributedSampler(
            rows,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            resume_micro_step=resume_micro_step,
        )

    _v2_train.NATIVE_SLAT_GENRECON_V2_VERSION = NATIVE_SLAT_NO_VGGT_VERSION
    _v2_train.NativeConditionSLatDataset = MixedNativeConditionSLatDataset
    _v2_train.ObjectBalancedDistributedSampler = balanced_sampler_factory
    _v2_train.validate_native_slat_genrecon_v2_checkpoint = (
        strict_checkpoint_validator
    )
    _v2_train.build_native_slat_genrecon_v2_components = strict_component_builder
    _v2_train.load_ss_evidence = load_no_vggt_ss_evidence
    _ss_evidence.NATIVE_SS_GENRECON_EVAL = NATIVE_SS_NO_VGGT_EVAL
    _v2_train.main()


if __name__ == "__main__":
    main()
