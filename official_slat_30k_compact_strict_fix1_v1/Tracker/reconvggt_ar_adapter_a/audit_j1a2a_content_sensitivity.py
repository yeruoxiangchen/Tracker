#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from reconvggt_ar_adapter_a.train_pointpose_ss_lora import PointPoseCacheDataset
from reconvggt_ar_adapter_a.train_stock_preserving_pointpose_bridge import (
    HardNegativeMiner,
    architecture_audit,
    build_models,
)


def write_markdown(path: Path, report: dict) -> None:
    audit = report["architecture_audit"]
    sensitivity = audit["untrained_content_sensitivity"]
    lines = [
        "# J1a.2-A Untrained Content Sensitivity Audit",
        "",
        f"- decision: `{'PASS' if report['passed'] else 'FAIL'}`",
        f"- correct uid: `{report['correct_uid']}`",
        f"- shuffled uid: `{report['shuffled_uid']}`",
        f"- physical token correct-shuffled RMS: "
        f"`{sensitivity['physical_token_correct_shuffled_rms']:.8f}`",
        f"- final correct equals stock: "
        f"`{sensitivity['final_correct_is_stock_at_zero_init']}`",
        f"- final shuffled equals stock: "
        f"`{sensitivity['final_shuffled_is_stock_at_zero_init']}`",
        f"- null-present condition exact: `{audit['null_present_condition_exact']}`",
        f"- physical-off condition exact: `{audit['hard_stock_route_exact']}`",
        "",
        "## Stage Diagnostics",
        "",
        "| stage | correct attended RMS | shuffled attended RMS | "
        "correct-shuffled RMS | cosine |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stage_name, row in sensitivity["stages"].items():
        lines.append(
            f"| {stage_name} | {row['correct_internal_attended_rms']:.8f} | "
            f"{row['shuffled_internal_attended_rms']:.8f} | "
            f"{row['attended_correct_shuffled_rms']:.8f} | "
            f"{row['attended_correct_shuffled_cosine']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit untrained J1a.2-A internal content sensitivity while the final "
            "zero-init condition remains exactly stock."
        )
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physical_hidden_dim", type=int, default=128)
    parser.add_argument("--content_fusion_dim", type=int, default=128)
    parser.add_argument("--content_fusion_heads", type=int, default=4)
    parser.add_argument("--fusion_stages", default="0,1")
    parser.add_argument("--fail_on_audit", action="store_true")
    args = parser.parse_args()

    if args.index < 0:
        raise ValueError("index must be non-negative")
    args.bridge_fusion_mode = "content_visual8"
    args.bridge_last_blocks = 1
    args.physical_heads = 8
    args.local_fusion_hidden_dim = 128
    args.gradient_checkpointing = False

    random.seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    dataset = PointPoseCacheDataset(args.cache_manifest, indices="all")
    if args.index >= len(dataset):
        raise IndexError(f"index {args.index} is outside dataset size {len(dataset)}")
    negative_miner = HardNegativeMiner(dataset)
    negative_index = negative_miner[int(args.index)]

    pipeline, model, model_summary = build_models(args, device)
    model.eval()
    audit = architecture_audit(
        pipeline,
        model,
        dataset[int(args.index)],
        device,
        negative_sample=dataset[negative_index],
        raise_on_failure=False,
    )
    report = {
        "format": "reconvggt.j1a2a.untrained_content_sensitivity.v1",
        "args": vars(args),
        "correct_uid": dataset[int(args.index)]["uid"],
        "shuffled_uid": dataset[negative_index]["uid"],
        "model": model_summary,
        "architecture_audit": audit,
        "passed": bool(audit["passed"]),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(output_dir / "report.md", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if args.fail_on_audit and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
