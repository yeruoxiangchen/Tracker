#!/usr/bin/env python3
"""Train official with-VGGT Native-SLat using the frozen v2 optimization loop."""

from __future__ import annotations

import os
import sys
from typing import Any

from pose_point_depth_mv import evaluate_native_ss_stock_slat_mesh as _ss_evidence
from pose_point_depth_mv import train_native_slat_genrecon as _train
from pose_point_depth_mv.native_slat_genrecon_with_vggt_official import (
    NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION,
    build_native_slat_official_with_vggt_components,
    validate_native_slat_official_with_vggt_checkpoint,
)
from pose_point_depth_mv.native_ss_genrecon_no_vggt import NATIVE_SS_NO_VGGT_EVAL
from pose_point_depth_mv.no_vggt_ss_evidence import (
    load_no_vggt_ss_evidence,
)
from pose_point_depth_mv.proobjaverse_official_slat_with_vggt_cache import (
    WITH_VGGT_CONTEXT_VERSION,
    WithVGGTNativeConditionSLatDataset,
)
from pose_point_depth_mv.train_native_slat_genrecon_no_vggt import (
    no_vggt_upstream_binding,
)


_base_initial_stock_audit = _train.initial_stock_audit


def _argument(name: str) -> str | None:
    values: list[str] = []
    prefix = f"{name}="
    for index, value in enumerate(sys.argv):
        if value == name:
            if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
                raise ValueError(f"{name} requires exactly one value")
            values.append(sys.argv[index + 1])
        elif value.startswith(prefix):
            values.append(value[len(prefix) :])
    if len(values) > 1:
        raise ValueError(f"duplicate frozen argument {name}: {values}")
    return values[0] if values else None


def _freeze_argument(name: str, expected: str) -> None:
    actual = _argument(name)
    if actual is None:
        sys.argv.extend((name, expected))
    elif actual != expected:
        try:
            equivalent = float(actual) == float(expected)
        except ValueError:
            equivalent = False
        if not equivalent:
            raise ValueError(
                f"official with-VGGT protocol freezes {name}={expected}, got={actual}"
            )


def _with_vggt_initial_stock_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    report = _base_initial_stock_audit(*args, **kwargs)
    report.update(
        {
            "reference_floor": "V0",
            "reference_definition": (
                "native ReconViaGen Stock SLat with identical slat_vggt_cond"
            ),
            "with_vggt_context_version": WITH_VGGT_CONTEXT_VERSION,
            "step0_v_equals_v0": bool(
                report["conditional_max_abs"] == 0.0
                and report["unconditional_max_abs"] == 0.0
            ),
            "vggt_camera_consumed": False,
        }
    )
    if not report["step0_v_equals_v0"]:
        raise RuntimeError(f"with-VGGT step-0 V/V0 equivalence failed: {report}")
    return report


def main() -> None:
    if _argument("--init_checkpoint"):
        raise ValueError(
            "official with-VGGT training is fresh-from-Stock only; "
            "--init_checkpoint is forbidden"
        )
    for name, expected in (
        ("--architecture", "v2"),
        ("--lora_rank", "8"),
        ("--lora_alpha", "16"),
        ("--condition_channels", "1024"),
        ("--new_lr", "0.0001"),
        ("--lora_lr", "0.00003"),
        ("--new_weight_decay", "0.01"),
        ("--adam_beta1", "0.9"),
        ("--adam_beta2", "0.95"),
        ("--grad_clip", "1.0"),
        ("--warmup_steps", "-1"),
        ("--warmup_ratio", "0.02"),
        ("--ema_decay", "0.9995"),
        ("--amp_dtype", "bf16"),
        ("--p_uncond", "0.1"),
        ("--t_logit_mean", "1.0"),
        ("--t_logit_std", "1.0"),
        ("--t_schedule", "uniform"),
        ("--min_condition_views", "1"),
        ("--max_condition_views", "8"),
        ("--stock_context_views", "all"),
        ("--indices", "all"),
        ("--seed", "42"),
    ):
        _freeze_argument(name, expected)
    if "--gradient_checkpointing" not in sys.argv:
        sys.argv.append("--gradient_checkpointing")
    help_requested = "--help" in sys.argv or "-h" in sys.argv
    if not help_requested:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        grad_accum_text = _argument("--grad_accum")
        grad_accum = 4 if grad_accum_text is None else int(grad_accum_text)
        if world_size * grad_accum != 8:
            raise ValueError(
                "official with-VGGT protocol freezes global effective batch=8: "
                f"world_size={world_size} grad_accum={grad_accum}"
            )

    _train.NativeConditionSLatDataset = WithVGGTNativeConditionSLatDataset
    _train.NATIVE_SLAT_GENRECON_V2_VERSION = (
        NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION
    )
    _train.validate_native_slat_genrecon_v2_checkpoint = (
        validate_native_slat_official_with_vggt_checkpoint
    )
    _train.build_native_slat_genrecon_v2_components = (
        build_native_slat_official_with_vggt_components
    )
    _train.initial_stock_audit = _with_vggt_initial_stock_audit
    _ss_evidence.NATIVE_SS_GENRECON_EVAL = NATIVE_SS_NO_VGGT_EVAL
    _train.load_ss_evidence = load_no_vggt_ss_evidence
    _train.upstream_binding = no_vggt_upstream_binding
    _train.main()


if __name__ == "__main__":
    main()
