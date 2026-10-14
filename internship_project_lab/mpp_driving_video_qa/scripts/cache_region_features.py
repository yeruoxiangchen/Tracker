#!/usr/bin/env python3
"""Cache only selected native-resolution crops with the frozen original MPP EVA.

Deduplication is by image content and exact crop pixels, not by question ID.
The full-frame cache is neither rewritten nor copied into each question's cache.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl
from mpp_video_qa.model import build_visual_cache_model
from mpp_video_qa.region_views import REGION_FEATURE_PROTOCOL, json_atomic, region_file


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--upstream-root", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--visual-checkpoint-root", default="")
    p.add_argument("--bridge-init", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--existing-policy", choices=("validate", "trust"), default="validate")
    p.add_argument("--max-regions", type=int, default=0)
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--max-txt-len", type=int, default=1536)
    a = p.parse_args()
    a.feature_stage, a.view = "eva", "full"
    return a


def encoder_identity(args):
    return {"feature_stage": "eva", "model": "eva_clip_g_224",
            "frozen_ln_vision_sha256_source": file_sha256(args.bridge_init),
            "processor": "lavis.BlipImageEvalProcessor_224_native_rgb_crop_v1",
            "storage_dtype": "bfloat16", "compute_dtype": args.dtype,
            "patch_shape": [257, 1408]}


def cache_matches(path, entry, identity):
    try:
        payload = torch.load(path, map_location="cpu")
        features = payload["features"]
        return (payload.get("protocol") == REGION_FEATURE_PROTOCOL and
                all(payload.get(k) == entry[k] for k in ("region_key", "image_sha256", "pixel_box")) and
                payload.get("encoder_identity") == identity and features.shape == (257, 1408) and
                features.dtype == torch.bfloat16 and bool(torch.isfinite(features).all()))
    except (OSError, KeyError, RuntimeError, EOFError, ValueError):
        return False


def main():
    args = parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    output = Path(args.cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    identity = encoder_identity(args)
    manifest_path = output / "cache_manifest.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        if old["encoder_identity"] != identity or old["catalog_sha256"] != file_sha256(args.catalog):
            raise ValueError("region cache belongs to another encoder or crop catalog; use a new directory")
    if rank == 0:
        json_atomic(manifest_path, {"protocol": REGION_FEATURE_PROTOCOL, "encoder_identity": identity,
                                   "catalog_sha256": file_sha256(args.catalog)})
    entries = load_jsonl(args.catalog)
    if len({r["region_key"] for r in entries}) != len(entries):
        raise ValueError("crop catalog must be deduplicated")
    if args.max_regions:
        entries = entries[:args.max_regions]
    assigned = entries[rank::world]
    stats = {"validated": 0, "trusted": 0, "generated": 0, "stale": 0}
    def progress(stage, completed, **extra):
        value = {"status": stage, "rank": rank, "world_size": world, "pid": os.getpid(),
                 "completed": completed, "assigned_total": len(assigned), "regions_total": len(entries),
                 "stats": stats.copy(), "elapsed_seconds": time.time() - started,
                 "updated_unix_seconds": time.time(), **extra}
        json_atomic(output / ("progress_rank_%d.json" % rank), value)
        print(json.dumps(value), flush=True)
    progress("validating_cache", 0)
    pending = []
    for index, entry in enumerate(assigned, 1):
        path = region_file(output, entry["region_key"])
        if path.is_file() and args.existing_policy == "trust":
            stats["trusted"] += 1
        elif path.is_file() and cache_matches(path, entry, identity):
            stats["validated"] += 1
        else:
            stats["stale"] += int(path.is_file())
            pending.append(entry)
        if index % 200 == 0:
            progress("validating_cache", index, pending=len(pending))
    distributed = world > 1
    if distributed:
        torch.distributed.init_process_group("nccl", timeout=timedelta(hours=12))
    # Only EVA is loaded, not the language checkpoint. Serial loading still
    # avoids competing large reads on the project's shared filesystem.
    visual = None
    for loading_rank in range(world):
        if rank == loading_rank and pending:
            progress("loading_eva", stats["validated"] + stats["trusted"])
            visual = build_visual_cache_model(args).to(device).eval().requires_grad_(False)
        if distributed:
            torch.distributed.barrier()
    if pending:
        from lavis.processors.blip_processors import BlipImageEvalProcessor
        processor = BlipImageEvalProcessor(image_size=224)
        visual_dtype = next(visual.visual_encoder.parameters()).dtype
    hashed = {}
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        crops = []
        try:
            for entry in batch:
                frame_path = entry["frame_path"]
                if frame_path not in hashed:
                    hashed[frame_path] = file_sha256(frame_path)
                if hashed[frame_path] != entry["image_sha256"]:
                    raise ValueError("source image changed after detection: %s" % frame_path)
                with Image.open(frame_path) as source:
                    rgb = source.convert("RGB")
                    try:
                        x1, y1, x2, y2 = entry["pixel_box"]
                        if not (0 <= x1 < x2 <= rgb.width and 0 <= y1 < y2 <= rgb.height):
                            raise ValueError("crop pixels outside original image")
                        crops.append(rgb.crop((x1, y1, x2, y2)))
                    finally:
                        rgb.close()
            pixels = torch.stack([processor(image) for image in crops]).to(device=device, dtype=visual_dtype)
            with torch.inference_mode():
                features = visual.encode_image(pixels, feature_stage="eva").to(torch.bfloat16).cpu()
            if features.shape[1:] != (257, 1408) or not torch.isfinite(features).all():
                raise RuntimeError("invalid region EVA output")
            for entry, feature in zip(batch, features):
                path = region_file(output, entry["region_key"])
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                # clone avoids serializing the storage of the entire crop batch.
                torch.save({"protocol": REGION_FEATURE_PROTOCOL, "encoder_identity": identity,
                            "region_key": entry["region_key"], "image_sha256": entry["image_sha256"],
                            "pixel_box": entry["pixel_box"], "features": feature.clone()}, temporary)
                temporary.replace(path)
                stats["generated"] += 1
        finally:
            for crop in crops:
                crop.close()
        if offset == 0 or offset // args.batch_size % 10 == 0 or offset + len(batch) == len(pending):
            progress("encoding_regions", stats["validated"] + stats["trusted"] + stats["generated"])
    totals = torch.tensor([stats[k] for k in ("validated", "trusted", "generated", "stale")], device=device)
    if distributed:
        torch.distributed.all_reduce(totals)
    progress("completed", len(assigned))
    if rank == 0:
        result = {"completed": True, "regions": len(entries), "catalog_regions": len(load_jsonl(args.catalog)),
                  "catalog_sha256": file_sha256(args.catalog), "encoder_identity": identity,
                  "stats": {k: int(v) for k, v in zip(("validated", "trusted", "generated", "stale"), totals)},
                  "elapsed_seconds": time.time() - started}
        json_atomic(output / "cache_summary.json", result)
        print(json.dumps(result, indent=2), flush=True)
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        args = parse_args()
        rank = int(os.environ.get("RANK", 0))
        json_atomic(Path(args.cache_dir) / ("failure_rank_%d.json" % rank),
                    {"status": "failed", "rank": rank, "updated_unix_seconds": time.time(),
                     "traceback": traceback.format_exc()})
        raise
