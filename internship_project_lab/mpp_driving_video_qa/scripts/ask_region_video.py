#!/usr/bin/env python3
"""Real short-clip inference with the same answer-free automatic view selector."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ask_video import read_uniform_frames
from mpp_video_qa.core import TASK_INSTRUCTIONS, dump_jsonl, file_sha256, stable_key
from mpp_video_qa.model import build_text_model, build_visual_cache_model, CachedVideoQA
from mpp_video_qa.region_views import PROPOSAL_PROTOCOL, json_atomic, plan_views, proposal_file, view_prompt
from mpp_video_qa.visual_bridge import add_bridge_arguments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_bridge_arguments(p)
    p.add_argument("--video", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--condition", choices=("full", "duplicate", "roi"), default="roi")
    p.add_argument("--detector-python", required=True)
    p.add_argument("--detector-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--detector-cache", default="")
    p.add_argument("--planner-manifest", required=True,
                   help="frozen training planner and detector settings, reused for user clips")
    p.add_argument("--adapter", required=True)
    p.add_argument("--projector", required=True)
    p.add_argument("--upstream-root", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--feature-stage", choices=("eva",), default="eva")
    p.add_argument("--task-instruction-version", choices=sorted(TASK_INSTRUCTIONS), default="legacy_v1")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--max-txt-len", type=int, default=1536)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=.05)
    p.add_argument("--lora-target-modules", nargs="+", default=["c_attn"])
    a = p.parse_args()
    if a.precision_protocol != "fp32_trainables_v1":
        p.error("use the same FP32-trained bridge/projector inference as SFT")
    # Application scope is a short driving clip, not sparse sampling of an
    # arbitrarily long recording. Export the native RGB observations losslessly.
    started = time.time()
    planner = json.loads(Path(a.planner_manifest).read_text())
    settings = planner["settings"]
    detector = planner["detector_identity"]
    request = detector["request"]
    if a.detector_model != request["model"]:
        p.error("inference detector differs from training planner")
    frames = read_uniform_frames(a.video, 5)
    segment = stable_key(str(Path(a.video).resolve()), Path(a.video).stat().st_mtime_ns)
    qid = stable_key(segment, a.question)
    destination = Path(a.output_root) / qid
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for index, image in enumerate(frames):
            path = destination / ("frame-%d.png" % index)
            if not path.is_file():
                image.save(path)
            paths.append(str(path.resolve()))
    finally:
        for image in frames:
            image.close()
    row = {"question_id": qid, "segment_id": segment, "question": a.question,
           "frames": paths, "answers": [], "source_subset": "user_clip"}
    dump_jsonl(destination / "input.jsonl", [row])
    env = dict(os.environ, HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
    # A single clip must not inherit torchrun's parent world/rank variables.
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        env.pop(key, None)
    if a.detector_cache:
        env["HF_HUB_CACHE"] = a.detector_cache
    command = [a.detector_python, str(ROOT / "scripts/propose_regions.py"),
               "--data", str(destination / "input.jsonl"), "--output-dir", str(destination / "proposals"),
               "--model", a.detector_model, "--box-threshold", str(request["box_threshold"]),
               "--text-threshold", str(request["text_threshold"]), "--max-candidates", str(request["max_candidates"])]
    revision = detector.get("resolved_revision") or request.get("revision")
    if revision:
        command.extend(("--revision", revision))
    if a.condition != "full":
        subprocess.run(command, env=env, check=True)
        proposal = json.loads(proposal_file(destination / "proposals", segment).read_text())
    else:
        from PIL import Image
        frame_metadata = []
        for path in paths:
            with Image.open(path) as source:
                frame_metadata.append({"path": path, "width": source.width, "height": source.height,
                                       "image_sha256": file_sha256(path), "candidates": []})
        proposal = {"protocol": PROPOSAL_PROTOCOL, "segment_id": segment,
                    "detector_identity": {"detector_used": False}, "frames": frame_metadata}
    plan = plan_views(qid, segment, a.question, paths, proposal,
                      settings["max_regions_per_frame"], settings["minimum_detection_score"], settings["context_padding"])
    dump_jsonl(destination / "views.jsonl", [plan])
    a.view = "full"
    visual = build_visual_cache_model(a).to("cuda").eval()
    from PIL import Image
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    processor = BlipImageEvalProcessor(image_size=224)
    dtype = next(visual.visual_encoder.parameters()).dtype
    features = []
    try:
        for view in plan["views"]:
            if a.condition == "full" and view["kind"] != "full":
                continue
            with Image.open(paths[view["frame_index"]]) as source:
                rgb = source.convert("RGB")
                try:
                    image = rgb.crop(tuple(view["pixel_box"])) if a.condition == "roi" and view["kind"] == "region" else rgb
                    try:
                        pixels = processor(image).unsqueeze(0).to("cuda", dtype=dtype)
                        with torch.inference_mode():
                            features.append(visual.encode_image(pixels, feature_stage="eva")[0].to(torch.bfloat16).cpu())
                    finally:
                        if image is not rgb:
                            image.close()
                finally:
                    rgb.close()
        tensor = torch.stack(features)
    finally:
        del visual
        if "pixels" in locals():
            del pixels
        gc.collect()
        torch.cuda.empty_cache()
    model = CachedVideoQA(build_text_model(a, train=False, adapter=a.adapter, projector=a.projector),
                          task_instruction=TASK_INSTRUCTIONS[a.task_instruction_version], feature_stage="eva").to("cuda").eval()
    prompt = view_prompt(a.question, plan, a.condition, TASK_INSTRUCTIONS[a.task_instruction_version])
    answer = model.generate(tensor.unsqueeze(0).to("cuda"), [a.question], a.max_new_tokens, prompts=[prompt])[0]
    value = {"question": a.question, "answer": answer, "condition": a.condition, "views": len(tensor),
             "detector_used": a.condition != "full",
             "view_plan": str((destination / "views.jsonl").resolve()), "plan_sha256": plan["plan_sha256"],
             "end_to_end_seconds_including_detection_and_loading": time.time() - started}
    json_atomic(destination / "answer.json", value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
