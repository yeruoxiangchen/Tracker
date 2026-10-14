#!/usr/bin/env python3
"""Prepare, monitor, and summarize the driving-perspective prompt ablation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import (DRIVING_PERSPECTIVE_TASK_INSTRUCTION,
                               TASK_INSTRUCTION, dump_jsonl, file_sha256,
                               load_jsonl, normalize_answer)

MODELS = ("base", "epoch03")
CONDITIONS = ("random_dev", "action_original", "action_rephrased")


def ai_self_reference(text):
    text = str(text).lower().strip()
    markers = ("as an ai", "i am currently analyzing", "i am providing an answer",
               "i am not capable of driving", "i cannot drive")
    return any(marker in text for marker in markers)


def action_prefix(answer):
    text = str(answer).lower().strip()
    for prefix, name in (("i am stopping", "stopping"),
                         ("i am remaining stopped", "remaining_stopped"),
                         ("i am accelerating", "accelerating"),
                         ("i am decelerating", "decelerating"),
                         ("i am starting", "starting"),
                         ("i am turning", "turning"),
                         ("i am following", "following"),
                         ("i am maintaining", "maintaining"),
                         ("i am continuing", "continuing")):
        if text.startswith(prefix):
            return name
    return "other_wording"


def prepare(args):
    source = load_jsonl(args.source_dir / "diagnostic.jsonl")
    rows = [r for r in source if r["diagnostic"]["condition"] in ("correct", "rephrased")]
    assert len(rows) == 596, len(rows)
    assert len({r["question_id"] for r in rows}) == len(rows)
    assert sum(r["diagnostic"]["random_dev_sample"] for r in rows) == 512
    assert sum(r["diagnostic"]["condition"] == "correct" and
               r["diagnostic"]["current_action"] for r in rows) == 48
    assert sum(r["diagnostic"]["condition"] == "rephrased" for r in rows) == 48
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.output_dir / "diagnostic.jsonl", rows)
    manifest = {
        "protocol": "driving_perspective_prompt_ablation_v1",
        "records_per_model": len(rows),
        "models": list(MODELS),
        "legacy_task_instruction": TASK_INSTRUCTION,
        "candidate_task_instruction": DRIVING_PERSPECTIVE_TASK_INSTRUCTION,
        "random_dev_records": 512,
        "action_original_records": 48,
        "action_rephrased_records": 48,
        "source_diagnostic_sha256": file_sha256(args.source_dir / "diagnostic.jsonl"),
        "diagnostic_sha256": file_sha256(args.output_dir / "diagnostic.jsonl"),
        "selection_rule": "Prompt chosen only from fixed dev controls; official eval remains untouched.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def group(rows, condition):
    if condition == "random_dev":
        return [r for r in rows if r["diagnostic"]["random_dev_sample"]]
    if condition == "action_original":
        return [r for r in rows if r["diagnostic"]["condition"] == "correct" and
                r["diagnostic"]["current_action"]]
    return [r for r in rows if r["diagnostic"]["condition"] == "rephrased"]


def metrics(rows):
    return {
        "records": len(rows),
        "judge_accuracy": statistics.mean(r["lingo_judge_correct"] for r in rows),
        "token_f1": statistics.mean(r["token_f1"] for r in rows),
        "mean_answer_words": statistics.mean(len(r["prediction"].split()) for r in rows),
        "ai_self_reference_count": sum(ai_self_reference(r["prediction"]) for r in rows),
        "stopping_prefix_count": sum(action_prefix(r["prediction"]) == "stopping" for r in rows),
        "prediction_prefixes": dict(Counter(action_prefix(r["prediction"]) for r in rows)),
    }


def compare(legacy, candidate):
    old = {r["question_id"]: r for r in legacy}
    new = {r["question_id"]: r for r in candidate}
    assert old.keys() == new.keys()
    keys = sorted(old)
    return {
        "records": len(keys),
        "fixed": sum(not old[k]["lingo_judge_correct"] and new[k]["lingo_judge_correct"]
                     for k in keys),
        "regressed": sum(old[k]["lingo_judge_correct"] and not new[k]["lingo_judge_correct"]
                         for k in keys),
        "both_correct": sum(old[k]["lingo_judge_correct"] and new[k]["lingo_judge_correct"]
                            for k in keys),
        "both_wrong": sum(not old[k]["lingo_judge_correct"] and
                          not new[k]["lingo_judge_correct"] for k in keys),
        "answer_changed": sum(normalize_answer(old[k]["prediction"]) !=
                              normalize_answer(new[k]["prediction"]) for k in keys),
        "ai_self_reference_before": sum(ai_self_reference(old[k]["prediction"]) for k in keys),
        "ai_self_reference_after": sum(ai_self_reference(new[k]["prediction"]) for k in keys),
    }


def report(args):
    old_root = args.source_dir
    expected_ids = {r["question_id"] for r in load_jsonl(args.output_dir / "diagnostic.jsonl")}
    result = {"protocol": "driving_perspective_prompt_ablation_v1", "models": {}}
    case_rows = []
    for model in MODELS:
        legacy_all = load_jsonl(old_root / model / "scored/scored_predictions.jsonl")
        legacy = [r for r in legacy_all if r["question_id"] in expected_ids]
        candidate = load_jsonl(args.output_dir / model / "scored/scored_predictions.jsonl")
        assert len(legacy) == len(candidate) == len(expected_ids)
        model_result = {"legacy": {}, "candidate": {}, "paired": {}}
        for condition in CONDITIONS:
            old_group, new_group = group(legacy, condition), group(candidate, condition)
            model_result["legacy"][condition] = metrics(old_group)
            model_result["candidate"][condition] = metrics(new_group)
            model_result["paired"][condition] = compare(old_group, new_group)
        result["models"][model] = model_result
        old_by_id = {r["question_id"]: r for r in legacy}
        for row in candidate:
            if row["diagnostic"]["current_action"]:
                case_rows.append({"model": model, "question_id": row["question_id"],
                                  "condition": row["diagnostic"]["condition"],
                                  "question": row["question"], "references": row["references"],
                                  "legacy_prediction": old_by_id[row["question_id"]]["prediction"],
                                  "candidate_prediction": row["prediction"],
                                  "legacy_judge_correct": old_by_id[row["question_id"]]["lingo_judge_correct"],
                                  "candidate_judge_correct": row["lingo_judge_correct"]})
    result["decision_rule"] = {
        "primary": "Adopt only if epoch03 random-dev Judge improves or is unchanged within one question, while action role errors decrease.",
        "secondary": "Inspect fixed/regressed action cases because Lingo-Judge has known false positives.",
        "test_set_policy": "Do not tune on official eval; run it once only after choosing the prompt on dev.",
    }
    result["limitations"] = [
        "This is an inference prompt ablation; existing SFT was trained with legacy_v1.",
        "Lingo-Judge is a text-only proxy and has observed false positives.",
        "Clarifying the speaker can fix role interpretation without improving visual action recognition.",
    ]
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    dump_jsonl(args.output_dir / "action_cases.jsonl", case_rows)
    lines = ["# 驾驶视角任务说明对照", "",
             "只修改任务说明；视频、问题、参考答案、模型权重和解码参数保持一致。", "",
             "| 模型 | 条件 | 旧说明 Judge | 新说明 Judge | fixed | regressed | AI 自述（前→后） |",
             "|---|---|---:|---:|---:|---:|---:|"]
    labels = {"random_dev": "随机 dev 512", "action_original": "动作原问法 48",
              "action_rephrased": "动作改写 48"}
    for model in MODELS:
        item = result["models"][model]
        for condition in CONDITIONS:
            old, new, pair = (item["legacy"][condition], item["candidate"][condition],
                              item["paired"][condition])
            lines.append("| %s | %s | %.2f%% | %.2f%% | %d | %d | %d→%d |" % (
                model, labels[condition], old["judge_accuracy"] * 100,
                new["judge_accuracy"] * 100, pair["fixed"], pair["regressed"],
                pair["ai_self_reference_before"], pair["ai_self_reference_after"]))
    lines += ["", "逐题动作对照见 `action_cases.jsonl`，完整字段见 `report.json`。"]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def status(args):
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    total = manifest["records_per_model"]
    result = {"total_per_model": total, "models": {}}
    for model in MODELS:
        path = args.output_dir / model / "predictions.jsonl"
        generated = 0
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    json.loads(line)
                    generated += 1
                except json.JSONDecodeError:
                    pass
        scored = args.output_dir / model / "scored/summary.json"
        result["models"][model] = {
            "generated": generated, "total": total,
            "stage": "scored" if scored.exists() else
                     ("scoring_pending_or_in_progress" if generated == total else
                      "generation_incomplete"),
            "seconds_since_prediction_update": (
                round(time.time() - path.stat().st_mtime, 1) if path.exists() else None),
        }
    result["report_ready"] = (args.output_dir / "report.json").exists()
    print(json.dumps(result, indent=2, ensure_ascii=False))


def wait_report(args):
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        ready = [m for m in MODELS if
                 (args.output_dir / m / "scored/summary.json").exists()]
        print(json.dumps({"stage": "waiting", "ready_models": ready,
                          "expected_models": list(MODELS)}), flush=True)
        if len(ready) == len(MODELS):
            report(args)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("prompt ablation did not finish within timeout")
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "status", "report", "wait-report"))
    parser.add_argument("--source-dir", type=Path,
                        default=ROOT / "outputs/sft_diagnosis_v1")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/task_instruction_ablation_v1")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    {"prepare": prepare, "status": status, "report": report,
     "wait-report": wait_report}[args.action](args)
