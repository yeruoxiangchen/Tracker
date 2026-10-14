#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpp_video_qa.core import PROTOCOL, load_jsonl
from mpp_video_qa.data import cache_file
from mpp_video_qa.observations import open_views, VIEW_PROTOCOL
from mpp_video_qa.model import (FEATURE_STAGE_PROJECTED, FEATURE_STAGE_QFORMER, FEATURE_STAGE_EVA,
                                FEATURE_STAGES, build_visual_cache_model)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--visual-checkpoint-root", default="",
                        help="optional root containing ckpt/blip2; MPP projection still comes "
                             "from --checkpoint-root")
    parser.add_argument("--bridge-init", default="")
    parser.add_argument("--view", choices=("full", "full_crop"), default="full")
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--max-segments", type=int, default=0,
                        help="optional global segment limit for smoke tests")
    parser.add_argument("--max-txt-len", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--existing-policy", choices=("validate", "trust", "rebuild"),
                        default="validate", help="how to handle an existing cache entry")
    parser.add_argument("--feature-stage", choices=FEATURE_STAGES,
                        default=FEATURE_STAGE_PROJECTED,
                        help="cache MPP projection output or its 768-d Q-Former input")
    return parser.parse_args()


def select_frames(paths, wanted):
    if not paths:
        raise ValueError("empty frame list")
    if wanted <= 0:
        raise ValueError("max-frames must be positive")
    if wanted == 1:
        indices = [0]
    elif len(paths) == 1:
        indices = [0] * wanted
    else:
        indices = [round(i * (len(paths) - 1) / (wanted - 1)) for i in range(wanted)]
    return [paths[i] for i in indices]


def cache_matches(path, segment_id, frame_paths, feature_stage, view="full"):
    try:
        payload = torch.load(path, map_location="cpu")
        features = payload.get("features")
        cached_stage = payload.get("feature_stage")
        if cached_stage is None and isinstance(features, torch.Tensor) and \
                features.ndim == 3 and features.shape[-1] == 4096:
            cached_stage = FEATURE_STAGE_PROJECTED
        expected_hidden = {FEATURE_STAGE_QFORMER: 768, FEATURE_STAGE_PROJECTED: 4096,
                           FEATURE_STAGE_EVA: 1408}[feature_stage]
        return (payload.get("protocol") == PROTOCOL and
                payload.get("segment_id") == segment_id and
                payload.get("frame_paths") == frame_paths and
                payload.get("view", "full") == view and
                cached_stage == feature_stage and
                isinstance(features, torch.Tensor) and features.ndim == 3 and
                features.shape[0] == len(frame_paths) * (2 if view == "full_crop" else 1) and
                features.shape[1] == (257 if feature_stage == FEATURE_STAGE_EVA else 32) and
                features.shape[2] == expected_hidden and
                torch.isfinite(features).all().item())
    except Exception:
        return False


def main():
    args = args_parser()
    rows = []
    for path in args.data:
        rows.extend(load_jsonl(path))
    unique = {}
    for row in rows:
        unique.setdefault(row["segment_id"], row)
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl", timeout=timedelta(hours=12))
    device = torch.device("cuda", local_rank)
    all_items = sorted(unique.items())
    if args.max_segments > 0:
        all_items = all_items[:args.max_segments]
    items = all_items[rank::world]
    pending = []
    stats = {"validated": 0, "trusted": 0, "stale": 0, "generated": 0}
    for segment_id, row in items:
        destination = cache_file(args.cache_dir, segment_id)
        paths = select_frames(row["frames"], args.max_frames)
        if destination.is_file() and args.existing_policy == "trust":
            stats["trusted"] += 1
        elif destination.is_file() and args.existing_policy == "validate" and \
                cache_matches(destination, segment_id, paths, args.feature_stage, args.view):
            stats["validated"] += 1
        else:
            if destination.is_file():
                stats["stale"] += 1
            pending.append((segment_id, row, paths))

    local_pending = torch.tensor([len(pending)], device=device, dtype=torch.long)
    if distributed:
        torch.distributed.all_reduce(local_pending)
    model = None
    if int(local_pending.item()) > 0:
        for loading_rank in range(world):
            if rank == loading_rank:
                model = build_visual_cache_model(args)
                model = model.to(device).eval()
            if distributed:
                torch.distributed.barrier()
        from lavis.processors.blip_processors import BlipImageEvalProcessor
        processor = BlipImageEvalProcessor(image_size=224)
    if not pending:
        # Other ranks may still generate entries; retain collective ordering below.
        visual_dtype = None
    else:
        visual_dtype = next(model.visual_encoder.parameters()).dtype
    started = time.time()
    for index, (segment_id, row, paths) in enumerate(pending, 1):
        destination = cache_file(args.cache_dir, segment_id)
        images = open_views(paths, args.view)
        try:
            pixels = torch.stack([processor(image) for image in images]).to(
                device=device, dtype=visual_dtype)
            with torch.inference_mode():
                features = model.encode_image(
                    pixels, feature_stage=args.feature_stage).to(torch.bfloat16).cpu()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            torch.save({"protocol": PROTOCOL, "feature_stage": args.feature_stage,
                        "segment_id": segment_id,
                        "view": args.view, "view_protocol": VIEW_PROTOCOL,
                        "frame_paths": paths, "features": features}, temporary)
            temporary.replace(destination)
            stats["generated"] += 1
        finally:
            for image in images:
                image.close()
        if index == 1 or index % 20 == 0 or index == len(pending):
            progress = {"rank": rank, "completed": index, "assigned_new": len(pending),
                        "assigned_total": len(items), "segments": len(all_items),
                        "stats": stats, "elapsed_seconds": time.time() - started}
            print(json.dumps(progress), flush=True)
            progress_path = Path(args.cache_dir) / ("progress_rank_%d.json" % rank)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    totals = torch.tensor([stats[key] for key in ("validated", "trusted", "stale", "generated")],
                          device=device, dtype=torch.long)
    if distributed:
        torch.distributed.all_reduce(totals)
        torch.distributed.barrier()
    if rank == 0:
        summary = {"completed": True, "segments": len(all_items),
                   "existing_policy": args.existing_policy,
                   "validated": int(totals[0]), "trusted": int(totals[1]),
                   "stale_rebuilt": int(totals[2]), "generated": int(totals[3]),
                   "max_frames": args.max_frames, "feature_stage": args.feature_stage,
                   "hidden_size": {FEATURE_STAGE_QFORMER: 768, FEATURE_STAGE_PROJECTED: 4096,
                                   FEATURE_STAGE_EVA: 1408}[args.feature_stage]}
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.cache_dir) / "cache_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
