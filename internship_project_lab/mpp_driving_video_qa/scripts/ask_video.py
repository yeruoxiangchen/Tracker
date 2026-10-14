#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import GenerationConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpp_video_qa.core import build_generation_prompt, build_user_prompt, TASK_INSTRUCTIONS
from mpp_video_qa.model import (build_full_visual_model, build_visual_cache_model,
                                build_text_model, CachedVideoQA)
from mpp_video_qa.visual_bridge import add_bridge_arguments


def read_uniform_frames(path, count):
    cap = cv2.VideoCapture(str(path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        raise RuntimeError("could not read video: %s" % path)
    indices = [round(i * (frame_count - 1) / (count - 1)) for i in range(count)] \
        if count > 1 else [frame_count // 2]
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            cap.release(); raise RuntimeError("decode failed at frame %d" % index)
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB"))
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser()
    add_bridge_arguments(parser)
    parser.add_argument("--video", required=True); parser.add_argument("--question", required=True)
    parser.add_argument("--adapter", default=""); parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--projector", default="")
    parser.add_argument("--upstream-root", required=True); parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--max-txt-len", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--lora-r", type=int, default=16); parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["c_attn"])
    parser.add_argument("--task-instruction-version", choices=sorted(TASK_INSTRUCTIONS), default="legacy_v1")
    args = parser.parse_args()
    if args.precision_protocol == "fp32_trainables_v1":
        # Use exactly the cached evaluation embedding/generation path, including
        # BF16 storage rounding at the frozen EVA boundary.
        args.feature_stage = "eva"
        args.view = "full"
        visual = build_visual_cache_model(args).to("cuda").eval()
        from lavis.processors.blip_processors import BlipImageEvalProcessor
        processor = BlipImageEvalProcessor(image_size=224)
        frames = read_uniform_frames(args.video, args.frames)
        try:
            pixels = torch.stack([processor(frame) for frame in frames]).to(
                "cuda", dtype=next(visual.visual_encoder.parameters()).dtype)
            with torch.inference_mode():
                features = visual.encode_image(pixels, feature_stage="eva").to(torch.bfloat16).cpu()
        finally:
            for frame in frames:
                frame.close()
        del visual, pixels
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        model = CachedVideoQA(
            build_text_model(args, train=False, adapter=args.adapter, projector=args.projector),
            task_instruction=TASK_INSTRUCTIONS[args.task_instruction_version],
            feature_stage="eva").to("cuda").eval()
        print(model.generate(features.unsqueeze(0).to("cuda"), [args.question])[0])
        return
    if args.qformer_mode == "full":
        parser.error("full Q-Former video inference requires --precision-protocol fp32_trainables_v1")
    model = build_full_visual_model(
        args, args.adapter, projector=args.projector).to("cuda").eval()
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    processor = BlipImageEvalProcessor(image_size=224)
    frames = read_uniform_frames(args.video, args.frames)
    dtype = next(model.visual_encoder.parameters()).dtype
    pixels = torch.stack([processor(frame) for frame in frames]).to("cuda", dtype=dtype)
    prompt = build_generation_prompt(build_user_prompt(args.question, args.frames))
    generation = GenerationConfig(max_new_tokens=96, do_sample=False,
                                  pad_token_id=model.llm_tokenizer.eod_id,
                                  eos_token_id=model.llm_tokenizer.im_end_id)
    with torch.inference_mode():
        answer = model.generate({"image": pixels, "text": [prompt]},
                                generation_config=generation)[0]
    print(answer)


if __name__ == "__main__":
    main()
