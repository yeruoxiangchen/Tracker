#!/usr/bin/env python3
"""Run the paper's frozen SS30K + SLat30K endpoint with bound assets."""

from __future__ import annotations

import argparse

from pose_aligned_reconstruction import infer_real_proobjaverse_official_ss_slat
from pose_aligned_reconstruction.current_30k import (
    ABC_R_BRIDGE,
    CHECKPOINT_STEP,
    PRETRAINED,
    SLAT30K_CHECKPOINT,
    SS30K_REPORT,
    STOCK_SLAT_FREEZE,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_input_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16"
    )
    parser.add_argument("--pretrained", default=PRETRAINED)
    parser.add_argument("--object", action="append")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.native_ss_report = str(SS30K_REPORT.path)
    args.native_slat_checkpoint = str(SLAT30K_CHECKPOINT.path)
    args.expected_slat_step = CHECKPOINT_STEP
    args.cross_deployment_bridge_report = str(ABC_R_BRIDGE.path)
    args.stock_slat_freeze = str(STOCK_SLAT_FREEZE.path)
    args.weights = "ema"
    infer_real_proobjaverse_official_ss_slat.run(args)


if __name__ == "__main__":
    main()
