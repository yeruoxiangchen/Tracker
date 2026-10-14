#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import best_reference_scores, load_jsonl


def load_judge(model_name, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    return tokenizer, model


@torch.inference_mode()
def judge_score(tokenizer, model, question, references, prediction):
    # Match wayveai/LingoQA benchmark/judge.py::compute/preprocess exactly.
    # The question is deliberately not normalized; retain the official 128 limit.
    references = [str(ref).lower().strip() for ref in references]
    prediction = str(prediction).lower().strip()
    texts = [f"{tokenizer.cls_token}\nQuestion: {question}\nAnswer: {ref}\nStudent: {prediction}"
             for ref in references]
    encoded = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(next(model.parameters()).device)
    logits = model(**encoded).logits.squeeze(-1)
    return float(logits.max()), bool(logits.max() > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lingo-judge", action="store_true")
    parser.add_argument("--judge-model", default="wayveai/Lingo-Judge")
    args = parser.parse_args()
    rows = load_jsonl(args.predictions)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    def progress(completed, status):
        payload = {"status": status, "completed": completed, "total": len(rows),
                   "updated_unix_seconds": time.time()}
        temporary = output / "score_progress.json.tmp"
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(output / "score_progress.json")
    progress(0, "loading_judge" if args.lingo_judge else "scoring")
    tokenizer = judge = None
    if args.lingo_judge:
        tokenizer, judge = load_judge(args.judge_model, torch.device("cuda"))
    details = []
    for row in rows:
        exact, f1 = best_reference_scores(row["prediction"], row["references"])
        detail = dict(row, normalized_exact=exact, token_f1=f1)
        if judge is not None:
            score, correct = judge_score(tokenizer, judge, row["question"],
                                         row["references"], row["prediction"])
            detail.update(lingo_judge_logit=score, lingo_judge_correct=correct)
            original_refs = row.get("diagnostic", {}).get("original_references")
            if original_refs:
                original_score, original_correct = judge_score(
                    tokenizer, judge, row["question"], original_refs, row["prediction"])
                detail.update(original_reference_judge_logit=original_score,
                              original_reference_judge_correct=original_correct)
        details.append(detail)
        if len(details) % 100 == 0:
            progress(len(details), "scoring")
    summary = {
        "judge_protocol": "official_lower_strip_maxref_logit_gt0_maxlength128_v1" if judge is not None else None,
        "records": len(details),
        "normalized_exact": statistics.mean(x["normalized_exact"] for x in details) if details else 0,
        "token_f1": statistics.mean(x["token_f1"] for x in details) if details else 0,
        "lingo_judge_accuracy": (statistics.mean(x["lingo_judge_correct"] for x in details)
                                  if judge is not None and details else None),
    }
    with open(output / "scored_predictions.jsonl", "w", encoding="utf-8") as handle:
        for row in details: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    progress(len(details), "completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
