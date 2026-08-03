#!/usr/bin/env python3
"""Freeze the exact pretrained Stock SLAT Flow and Mesh decoder identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pose_point_depth_mv.native_slat_genrecon import make_stock_slat_freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Stock-SLAT freeze requires one CUDA GPU because the upstream "
            "FlexiCubes decoder creates CUDA tables during construction"
        )
    from pose_point_depth_mv.train_direct_flow import install_unused_model_stubs
    from trellis.pipelines import TrellisVGGTTo3DPipeline

    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    flow = pipeline.models["slat_flow_model"].to(device).eval()
    decoder = pipeline.models["slat_decoder_mesh"].to(device).eval()
    for module in (flow, decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    payload = make_stock_slat_freeze(
        pretrained=args.pretrained,
        flow=flow,
        decoder=decoder,
        sampler_params=dict(pipeline.slat_sampler_params),
        normalization=dict(pipeline.slat_normalization),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": True, "output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
