#!/usr/bin/env python3
"""Local HF visual teacher. Run in a modern Transformers environment, not MPP's.

Only chronological images and the question enter the prompt. References are
written to the output for subsequent scoring, never provided to the teacher.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import TASK_INSTRUCTIONS, file_sha256, load_jsonl


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def model_identity(path):
    root = Path(path).resolve()
    if not (root / "config.json").is_file():
        raise ValueError("--model must name a downloaded local HF model directory")
    # Avoid re-reading multi-GB weights on shared storage at every invocation.
    # This is explicitly a file stat identity, not a content hash of all weights.
    weights = sorted(list(root.glob("*.safetensors")) + list(root.glob("pytorch_model*.bin")))
    if not weights:
        raise ValueError("local teacher weights are missing")
    return {"path": str(root), "metadata_sha256": {
        str(p.relative_to(root)): file_sha256(p) for p in sorted(root.glob("*.json"))},
        "weight_stat_identity": [{"file": p.name, "bytes": p.stat().st_size,
                                  "mtime_ns": p.stat().st_mtime_ns} for p in weights]}


def messages_for(row, images, instruction):
    content = []
    for index, picture in enumerate(images):
        content.extend([{"type": "text", "text": "Frame %d, chronological order:" % (index + 1)},
                        {"type": "image", "image": picture}])
    content.append({"type": "text", "text": instruction + "\nQuestion: " + row["question"]})
    return [{"role": "user", "content": content}]


def build_processor(model_path, min_pixels, max_pixels, input_mode="multi_image"):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    # AutoProcessor in Transformers 4.57.6 does not forward top-level
    # min_pixels/max_pixels to Qwen3VL's image processor. Set the actual image
    # processor budget explicitly; otherwise its default is 16M pixels/image.
    image_processor = processor.image_processor
    if not {"shortest_edge", "longest_edge"} <= set(image_processor.size):
        raise ValueError("teacher image processor needs a model-specific pixel-budget adapter")
    image_processor.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
    image_processor.min_pixels = min_pixels
    image_processor.max_pixels = max_pixels
    if input_mode == "video":
        # Pad five source frames to a complete temporal block BEFORE resizing.
        # The installed video smart_resize rounds odd T before checking the
        # total budget. Padding first makes H/W match the image-mode budget.
        video_processor = processor.video_processor
        factor = video_processor.temporal_patch_size
        encoded_frames = math.ceil(5 / factor) * factor
        video_processor.size = {"shortest_edge": encoded_frames * min_pixels,
                                "longest_edge": encoded_frames * max_pixels}
    return processor


def build_teacher_inputs(row, images, processor, instruction, input_mode="multi_image",
                         frame_interval_seconds=1.0):
    """Return HF inputs and actual observation metadata, never reference text."""
    if input_mode == "multi_image":
        messages = messages_for(row, images, instruction)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt")
        grid = inputs["image_grid_thw"]
        merge_size = processor.image_processor.merge_size
        metadata = {"source_frames": len(images), "input_mode": input_mode}
    elif input_mode == "video":
        import numpy as np
        import torch
        if len(images) != 5 or not math.isfinite(frame_interval_seconds) or frame_interval_seconds <= 0:
            raise ValueError("video mode requires five frames and a positive finite frame interval")
        frames = torch.from_numpy(np.stack([np.asarray(im) for im in images])).permute(0, 3, 1, 2)
        temporal = processor.video_processor.temporal_patch_size
        padding = (-len(images)) % temporal
        if padding:
            frames = torch.cat([frames, frames[-1:].repeat(padding, 1, 1, 1)], dim=0)
        indices = list(range(len(images))) + [len(images) - 1] * padding
        video_metadata = [{"total_num_frames": len(indices), "fps": 1.0 / frame_interval_seconds,
                           "frames_indices": indices}]
        messages = [{"role": "user", "content": [
            {"type": "video", "video": row["frames"]},
            {"type": "text", "text": instruction + "\nQuestion: " + row["question"]}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[frames], video_metadata=video_metadata,
                           do_sample_frames=False, return_tensors="pt")
        grid = inputs["video_grid_thw"]
        merge_size = processor.video_processor.merge_size
        metadata = {"source_frames": len(images), "encoded_frames": len(indices),
                    "input_mode": input_mode, "frame_interval_seconds": frame_interval_seconds,
                    "frame_indices": indices, "source_timestamps_seconds": [i * frame_interval_seconds for i in range(5)],
                    "padding": "repeat_last_frame", "resampled": False}
    else:
        raise ValueError("unknown teacher input mode: %s" % input_mode)
    metadata.update(grid_thw=grid.tolist(), visual_tokens=int(grid.prod(dim=-1).sum().item()) // merge_size**2)
    return inputs, metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-records", type=int, default=0, help="bounded integration check; 0 processes every row")
    p.add_argument("--min-pixels", type=int, default=50176)
    p.add_argument("--max-pixels", type=int, default=1003520)
    p.add_argument("--task-instruction-version", choices=sorted(TASK_INSTRUCTIONS), default="legacy_v1")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--input-mode", choices=("multi_image", "video"), default="multi_image")
    p.add_argument("--frame-interval-seconds", type=float, default=1.0,
                   help="relative sampling interval; LingoQA's documented five frames over four seconds")
    args = p.parse_args()
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    rows = load_jsonl(args.data)
    if args.max_records > 0:
        rows = rows[:args.max_records]
    if not rows or len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("teacher data must have unique question IDs and be nonempty")
    if any(len(r["frames"]) != 5 for r in rows):
        raise ValueError("this experiment uses the original five LingoQA frames")
    if not 0 < args.min_pixels <= args.max_pixels:
        raise ValueError("invalid visual pixel budget")
    if not math.isfinite(args.frame_interval_seconds) or args.frame_interval_seconds <= 0:
        raise ValueError("frame interval must be positive and finite")
    output = args.output_dir
    identity = {"protocol": "five_frame_external_teacher_v1", "backend": "hf_vlm",
                "model": model_identity(args.model), "data_sha256": file_sha256(args.data),
                "view": "full", "teacher_context": "view", "references_in_prompt": False,
                "task_instruction_version": args.task_instruction_version,
                "task_instruction": TASK_INSTRUCTIONS[args.task_instruction_version],
                "max_new_tokens": args.max_new_tokens, "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels, "decoding": "greedy", "dtype": "bf16",
                "max_records": args.max_records,
                "transformers_version": transformers.__version__}
    # Preserve resumability of the completed legacy A experiment.
    if args.input_mode != "multi_image" or args.task_instruction_version != "legacy_v1":
        identity.update(protocol="five_frame_external_teacher_v2", input_mode=args.input_mode,
                        spatial_budget="per_source_frame", max_source_frames=5)
        if args.input_mode == "video":
            identity.update(frame_interval_seconds=args.frame_interval_seconds,
                            timing_source="LingoQA paper: five frames over four seconds; relative uniform sampling",
                            temporal_padding="repeat_last_before_resize", do_sample_frames=False)
    manifest = output / "run_manifest.json"
    predictions = output / "predictions.jsonl"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("teacher/data/settings changed; use a new output directory")
    if predictions.exists() and not manifest.exists():
        raise ValueError("predictions exist without their generation identity")
    write_json(manifest, identity)
    existing = load_jsonl(predictions) if predictions.exists() else []
    expected = {r["question_id"]: r for r in rows}
    done = {r["question_id"] for r in existing}
    if len(done) != len(existing) or not done <= set(expected):
        raise ValueError("unexpected or duplicate completed predictions")
    for r in existing:
        original = expected[r["question_id"]]
        if (r["segment_id"], r["question"], r["references"]) != (
                original["segment_id"], original["question"], original["answers"]):
            raise ValueError("completed prediction differs from source")
    started = time.monotonic()
    durations = []
    def progress(status):
        recent = sum(durations[-20:]) / len(durations[-20:]) if durations else None
        value = {"status": status, "completed": len(done), "total": len(rows),
                 "updated_unix_seconds": time.time(), "elapsed_seconds": time.monotonic() - started,
                 "recent_seconds_per_question": recent,
                 "estimated_remaining_seconds": (len(rows) - len(done)) * recent if recent is not None else None}
        write_json(output / "progress.json", value)
        print(json.dumps(value), flush=True)
        return value
    if len(done) == len(rows):
        write_json(output / "generation_summary.json", progress("completed"))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process; run the teacher in a GPU-enabled environment")
    progress("loading_teacher")
    processor = build_processor(args.model, args.min_pixels, args.max_pixels, args.input_mode)
    model = AutoModelForImageTextToText.from_pretrained(str(args.model), local_files_only=True,
        torch_dtype=torch.bfloat16, device_map=args.device_map).eval()
    started = time.monotonic()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    with predictions.open("a", encoding="utf-8") as stream:
        for row in rows:
            if row["question_id"] in done:
                continue
            question_started = time.monotonic()
            images = []
            try:
                for path in row["frames"]:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                inputs, observation = build_teacher_inputs(row, images, processor, identity["task_instruction"],
                                                          args.input_mode, args.frame_interval_seconds)
                inputs = inputs.to(model.device)
                with torch.inference_mode():
                    tokens = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                completion = tokens[0, inputs.input_ids.shape[1]:]
                prediction = processor.decode(completion, skip_special_tokens=True).strip()
                finished = any(t in eos_ids for t in completion.tolist())
                result = {"question_id": row["question_id"], "segment_id": row["segment_id"],
                          "question": row["question"], "references": row["answers"],
                          "source_subset": row["source_subset"], "prediction": prediction,
                          "finished": finished, "completion_tokens": len(completion),
                          "view": "full", "teacher_context": "view", "observation": observation,
                          "latency_seconds": time.monotonic() - question_started}
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                done.add(row["question_id"])
                durations.append(time.monotonic() - question_started)
                progress("generating")
            finally:
                for image in images:
                    image.close()
    write_json(output / "generation_summary.json", progress("completed"))


if __name__ == "__main__":
    main()
