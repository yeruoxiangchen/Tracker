#!/usr/bin/env python3
"""Generate matched teacher probes (MPP or a local HF VLM), then report scored pairs."""
import argparse
import base64
import html
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, file_sha256, TASK_INSTRUCTION, TASK_INSTRUCTIONS


def report(args):
    cases = load_jsonl(args.data)
    ids = {row["question_id"] for row in cases}
    predictions = {}
    for entry in args.scored:
        name, path = entry.split("=", 1)
        rows = load_jsonl(path)
        keyed = {row["question_id"]: row for row in rows}
        if len(keyed) != len(rows) or set(keyed) != ids:
            raise ValueError("incomplete/duplicate teacher comparison: %s" % name)
        for case in cases:
            pred = keyed[case["question_id"]]
            if pred["question"] != case["question"] or pred["references"] != case["answers"]:
                raise ValueError("teacher question/references differ")
        predictions[name] = keyed
    baseline = predictions[args.baseline]
    groups = {"all": cases}
    for name in sorted({row.get("diagnostic_group", "unclassified") for row in cases}):
        groups[name] = [r for r in cases if r.get("diagnostic_group", "unclassified") == name]
    result = {"records": len(cases), "data_sha256": file_sha256(args.data), "groups": {},
              "interpretation": "Judge is text-only. Review images; this selected probe is not population accuracy."}
    for group, rows in groups.items():
        result["groups"][group] = {}
        for name, keyed in predictions.items():
            pairs = [(bool(baseline[r["question_id"]]["lingo_judge_correct"]),
                      bool(keyed[r["question_id"]]["lingo_judge_correct"])) for r in rows]
            fixed = sum(not a and b for a, b in pairs)
            regressed = sum(a and not b for a, b in pairs)
            result["groups"][group][name] = {
                "records": len(rows), "correct": sum(b for _, b in pairs),
                "accuracy": sum(b for _, b in pairs) / len(rows),
                "fixed": fixed, "regressed": regressed, "net": fixed - regressed}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    parts = ['<!doctype html><meta charset="utf-8"><title>Teacher comparison</title>',
             '<style>body{font-family:sans-serif;max-width:1400px;margin:auto}img{width:19%}td{padding:8px;vertical-align:top}table{width:100%;border-collapse:collapse}td{border:1px solid #ccc}</style>',
             '<h1>Teacher comparison — selected dev cases</h1>',
             '<p>Review action, justification and visual support. Judge correctness is not a visual annotation.</p>']
    from PIL import Image
    for index, case in enumerate(cases, 1):
        parts.append('<h2>%d. %s</h2>' % (index, html.escape(case["question"])))
        parts.append('<p>%s · %s</p>' % (html.escape(case["segment_id"]),
                                          html.escape(case.get("diagnostic_group", ""))))
        for path in case["frames"]:
            with Image.open(path) as original:
                picture = original.convert("RGB")
                picture.thumbnail((480, 320))
                buffer = io.BytesIO()
                picture.save(buffer, format="JPEG", quality=80)
                picture.close()
            parts.append('<img src="data:image/jpeg;base64,%s">' % base64.b64encode(buffer.getvalue()).decode())
        parts.append('<p>Reference: %s</p><table>' % html.escape(" / ".join(case["answers"])))
        for name, keyed in predictions.items():
            row = keyed[case["question_id"]]
            parts.append('<tr><td>%s</td><td>%s</td><td>Judge: %s</td></tr>' %
                         (html.escape(name), html.escape(row["prediction"]), row["lingo_judge_correct"]))
        parts.append('</table>')
    (output / "review.html").write_text("\n".join(parts), encoding="utf-8")
    (output / "report.md").write_text("# Teacher comparison\n\n```json\n" +
                                      json.dumps(result, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def generate(args):
    import torch
    from mpp_video_qa.observations import open_views, teacher_prompt, VIEW_PROTOCOL, ANSWER_HINT_PROTOCOL
    from mpp_video_qa.opd import sample_completions
    from scripts.train_sft import seed_all
    from scripts.generate_answers import directory_fingerprint, write_json_atomic
    rows = load_jsonl(args.data)
    if args.num_samples < 1 or (args.num_samples > 1 and not args.sample):
        raise ValueError("multiple samples require stochastic generation")
    if (args.teacher_context == "answer_hint" or args.sample) and args.backend != "mpp":
        raise ValueError("answer-hint sampling currently supports the MPP backend")
    if args.teacher_context == "answer_hint" and any(r.get("opd_training_source") != "train" for r in rows):
        raise ValueError("answer-hint probes must be explicitly prepared from training data")
    if args.teacher_context == "answer_hint" and args.view != "full":
        raise ValueError("answer-hint probes require full frames")
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate source question IDs")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"data_sha256": file_sha256(args.data), "backend": args.backend,
                "view": args.view, "view_protocol": VIEW_PROTOCOL, "max_new_tokens": args.max_new_tokens,
                "model": args.model, "adapter": directory_fingerprint(args.adapter) if args.adapter else None,
                "projector": file_sha256(args.projector) if args.projector else None,
                "qformer": file_sha256(args.qformer_adapter) if args.qformer_adapter else None,
                "feature_stage": args.feature_stage, "cache_dir": str(Path(args.cache_dir).resolve()),
                "bridge_init": file_sha256(args.bridge_init) if args.bridge_init else None,
                "checkpoint_root": str(Path(args.checkpoint_root).resolve())}
    if args.sample or args.teacher_context != "view":
        identity.update(teacher_context=args.teacher_context, num_samples=args.num_samples,
                        sampling=args.sample, seed=args.seed,
                        sampling_distribution="temperature=1,top_p=1,top_k=0" if args.sample else "greedy",
                        answer_hint_protocol=ANSWER_HINT_PROTOCOL if args.teacher_context == "answer_hint" else None,
                        task_instruction_version=args.task_instruction_version)
    manifest = output / "run_manifest.json"
    destination = output / "predictions.jsonl"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("teacher run identity changed; use a new output directory")
    if destination.exists() and not manifest.exists():
        raise ValueError("existing predictions have no identity manifest")
    write_json_atomic(manifest, identity)
    existing = load_jsonl(destination) if destination.exists() else []
    done = {(r["question_id"], r.get("sample_id", 0)) for r in existing}
    expected = {(r["question_id"], k) for r in rows for k in range(args.num_samples)}
    if len(done) != len(existing) or not done <= expected:
        raise ValueError("unexpected/duplicate completed predictions")
    if len(done) == len(expected):
        summary = {"completed": len(done), "total": len(expected)}
        write_json_atomic(output / "generation_summary.json", summary)
        print(json.dumps(summary))
        return
    if args.backend == "mpp":
        from mpp_video_qa.data import CachedQADataset, cache_file
        from mpp_video_qa.model import build_text_model, CachedVideoQA
        dataset = CachedQADataset(args.data, args.cache_dir, args.feature_stage)
        model = CachedVideoQA(build_text_model(args, train=False, adapter=args.adapter,
                                               projector=args.projector),
                              task_instruction=TASK_INSTRUCTIONS[args.task_instruction_version],
                              feature_stage=args.feature_stage).to("cuda").eval()
    else:
        # Run this backend in a modern Transformers environment, isolated from MPP's py3.8 stack.
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    started = time.time()
    with destination.open("a", encoding="utf-8") as stream:
        for index, sample_id in ((i, k) for i in range(len(rows)) for k in range(args.num_samples)):
            row = rows[index]
            if (row["question_id"], sample_id) in done:
                continue
            if args.backend == "mpp":
                payload = torch.load(cache_file(args.cache_dir, row["segment_id"]), map_location="cpu")
                if payload.get("view", "full") != args.view or payload["frame_paths"] != row["frames"]:
                    raise ValueError("cache view/frame identity differs")
                features = dataset[index]["features"].unsqueeze(0).to("cuda")
                prompt = teacher_prompt(row["question"], len(row["frames"]), row["answers"],
                                        args.teacher_context, args.view,
                                        TASK_INSTRUCTIONS[args.task_instruction_version])
                finished = True
                if args.sample:
                    # Stable per-draw seeds make interrupted probes reproducible.
                    seed_all(args.seed + index * args.num_samples + sample_id)
                    completion, finished = sample_completions(
                        model, features, row["question"], args.max_new_tokens, prompt)
                    prediction = model.backbone.llm_tokenizer.decode(completion, skip_special_tokens=True).strip()
                else:
                    prediction = model.generate(features, [row["question"]], args.max_new_tokens,
                                                prompts=[prompt])[0]
            else:
                images = open_views(row["frames"], args.view)
                try:
                    content = []
                    stride = 2 if args.view == "full_crop" else 1
                    for i, image in enumerate(images):
                        label = "Time %d, %s" % (i // stride,
                                 "detail of same frame" if stride == 2 and i % 2 else "full frame")
                        content.extend([{"type": "text", "text": label}, {"type": "image", "image": image}])
                    content.append({"type": "text", "text": TASK_INSTRUCTION + "\nQuestion: " + row["question"]})
                    messages = [{"role": "user", "content": content}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                    prediction = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:],
                                                        skip_special_tokens=True)[0].strip()
                finally:
                    for image in images:
                        image.close()
            result = {"question_id": row["question_id"], "segment_id": row["segment_id"],
                      "question": row["question"], "references": row["answers"],
                      "source_subset": row.get("source_subset"), "prediction": prediction,
                      "diagnostic_group": row.get("diagnostic_group"), "view": args.view}
            if args.sample or args.teacher_context != "view":
                result.update(sample_id=sample_id, finished=finished,
                              teacher_context=args.teacher_context)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            done.add((row["question_id"], sample_id))
            progress = {"status": "generating", "completed": len(done), "total": len(expected),
                        "elapsed_seconds": time.time() - started}
            write_json_atomic(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    write_json_atomic(output / "generation_summary.json", {"completed": len(done), "total": len(expected)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("generate", "report"))
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backend", choices=("mpp", "hf_vlm"), default="mpp")
    p.add_argument("--model", default="", help="local HF VLM path for the hf_vlm backend")
    p.add_argument("--view", choices=("full", "full_crop"), default="full")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--upstream-root", default="")
    p.add_argument("--checkpoint-root", default="")
    p.add_argument("--adapter", default="")
    p.add_argument("--projector", default="")
    p.add_argument("--feature-stage", choices=("qformer", "eva"), default="eva")
    p.add_argument("--max-txt-len", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--scored", nargs="+", default=[], help="NAME=scored_predictions.jsonl")
    p.add_argument("--baseline", default="joint")
    p.add_argument("--teacher-context", choices=("view", "answer_hint"), default="view")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task-instruction-version", choices=sorted(TASK_INSTRUCTIONS), default="legacy_v1")
    from mpp_video_qa.visual_bridge import add_bridge_arguments
    add_bridge_arguments(p)
    args = p.parse_args()
    if args.action == "report":
        report(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
