#!/usr/bin/env python3
"""Detect image-only candidates in an independent modern Transformers environment.

Each torchrun worker owns disjoint videos. There is no NCCL training collective.
Reference answers and questions are never provided to the detector.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl, stable_key
from mpp_video_qa.region_views import (DETECTOR_LABELS, PROPOSAL_PROTOCOL,
                                      json_atomic, normalized_box, proposal_file)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", action="append", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--revision", default=None)
    p.add_argument("--box-threshold", type=float, default=0.30)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--max-candidates", type=int, default=30)
    p.add_argument("--max-segments", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def unique_segments(data_paths):
    unique = {}
    for path in data_paths:
        for row in load_jsonl(path):
            segment = row["segment_id"]
            paths = row["frames"]
            if segment in unique and unique[segment] != paths:
                raise ValueError("one segment has different ordered frames: %s" % segment)
            unique[segment] = paths
    return sorted(unique.items())


def source_stat(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cached_proposal_valid(path, segment, frames, request):
    if not path.is_file():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    return (value.get("protocol") == PROPOSAL_PROTOCOL and value.get("segment_id") == segment
            and value["detector_identity"]["request"] == request
            and [f["path"] for f in value["frames"]] == frames
            and all(f.get("source_stat") == source_stat(f["path"]) for f in value["frames"]))


def detect_frame(model, processor, image, args, device):
    import torch
    inputs = processor(images=image, text=". ".join(DETECTOR_LABELS) + ".",
                       return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    postprocess = processor.post_process_grounded_object_detection
    parameters = inspect.signature(postprocess).parameters
    # Transformers renamed box_threshold to threshold; do not mix the two APIs.
    options = {"text_threshold": args.text_threshold,
               "target_sizes": [(image.height, image.width)]}
    options["box_threshold" if "box_threshold" in parameters else "threshold"] = args.box_threshold
    result = postprocess(outputs, inputs.input_ids, **options)[0]
    labels = result.get("text_labels", result.get("labels"))
    if labels is None or len(labels) != len(result["boxes"]):
        raise RuntimeError("grounded detection returned no matching text labels")
    candidates = []
    for label, score, box in zip(labels, result["scores"].tolist(), result["boxes"].tolist()):
        if not isinstance(label, str):
            raise RuntimeError("text labels required, not integer class IDs")
        try:
            normalized = normalized_box(box, image.width, image.height)
        except ValueError:
            continue
        candidates.append({"label": label.strip().lower(), "score": float(score), "box": normalized})
    return sorted(candidates, key=lambda c: (-c["score"], c["label"], c["box"]))[:args.max_candidates]


def main():
    args = parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    request = {"model": args.model, "revision": args.revision, "labels": list(DETECTOR_LABELS),
               "box_threshold": args.box_threshold, "text_threshold": args.text_threshold,
               "max_candidates": args.max_candidates}
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if previous and previous["request"] != request:
        raise ValueError("proposal directory belongs to different detector settings; use a new directory")
    segments = unique_segments(args.data)
    if args.max_segments:
        segments = segments[:args.max_segments]
    assigned = segments[rank::world]
    started = time.time()
    progress_path = output / ("progress_rank_%d.json" % rank)
    def progress(stage, completed, **extra):
        value = {"status": stage, "rank": rank, "world_size": world, "pid": os.getpid(),
                 "completed": completed, "assigned_total": len(assigned),
                 "segments_total": len(segments), "updated_unix_seconds": time.time(),
                 "elapsed_seconds": time.time() - started, **extra}
        json_atomic(progress_path, value)
        print(json.dumps(value), flush=True)
    progress("validating_proposals", 0)
    pending = []
    for index, (segment, paths) in enumerate(assigned, 1):
        if not cached_proposal_valid(proposal_file(output, segment), segment, paths, request):
            pending.append((segment, paths))
        if index % 100 == 0:
            progress("validating_proposals", index, pending=len(pending))
    identity = previous.get("detector_identity") if previous else None
    model = processor = None
    if pending:
        progress("loading_detector", len(assigned) - len(pending), pending=len(pending))
        processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model, revision=args.revision)
        model = model.to(device).eval().requires_grad_(False)
        identity = {"request": request, "resolved_revision": getattr(model.config, "_commit_hash", None),
                    "config_sha256": stable_key(json.dumps(model.config.to_dict(), sort_keys=True)),
                    "processor_sha256": stable_key(json.dumps(processor.to_dict(), sort_keys=True)),
                    "transformers_version": __import__("transformers").__version__}
        if previous and identity != previous["detector_identity"]:
            raise ValueError("detector weights/processor changed; use a new proposal directory")
    if rank == 0:
        json_atomic(manifest_path, {"protocol": PROPOSAL_PROTOCOL, "request": request,
                    "detector_identity": identity, "data_sha256": {str(p): file_sha256(p) for p in args.data},
                    "segments": len(segments), "world_size": world})
    validated = len(assigned) - len(pending)
    for index, (segment, paths) in enumerate(pending, 1):
        frames = []
        for frame_index, path in enumerate(paths):
            image_hash = file_sha256(path)
            with Image.open(path) as source:
                image = source.convert("RGB")
                try:
                    candidates = detect_frame(model, processor, image, args, device)
                    for candidate in candidates:
                        candidate["candidate_id"] = stable_key(image_hash, candidate["label"], candidate["box"])
                    frames.append({"path": path, "image_sha256": image_hash, "source_stat": source_stat(path),
                                   "width": image.width, "height": image.height, "frame_index": frame_index,
                                   "candidates": candidates})
                finally:
                    image.close()
        json_atomic(proposal_file(output, segment), {"protocol": PROPOSAL_PROTOCOL, "segment_id": segment,
                    "detector_identity": identity, "frames": frames})
        if index == 1 or index % 10 == 0 or index == len(pending):
            progress("detecting", validated + index, validated=validated, generated=index)
    progress("completed", len(assigned), validated=validated, generated=len(pending))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        args = parse_args()
        rank = int(os.environ.get("RANK", 0))
        json_atomic(Path(args.output_dir) / ("failure_rank_%d.json" % rank),
                    {"status": "failed", "rank": rank, "updated_unix_seconds": time.time(),
                     "traceback": traceback.format_exc()})
        raise
