#!/usr/bin/env python3
"""Prepare fixed dev controls and report SFT diagnostics without changing training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import dump_jsonl, file_sha256, load_jsonl, normalize_answer

ACTION_QUESTION = 'What is the current action and its justification? Answer in the form "action, justification".'
MODELS = ("base", "epoch01", "epoch02", "epoch03")


def current_action(row):
    return "current action" in row["question"].lower()


def action_prefix(answer):
    """Descriptive lexical buckets, NOT ground-truth action labels."""
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
    train = load_jsonl(args.data_dir / "train.jsonl")
    dev = load_jsonl(args.data_dir / "dev.jsonl")
    train_segments = {r["segment_id"] for r in train}
    assert not train_segments & {r["segment_id"] for r in dev}
    sample = random.Random(args.seed).sample(sorted(dev, key=lambda r: r["question_id"]),
                                             min(args.sample_size, len(dev)))
    sampled = {r["question_id"] for r in sample}
    actions = sorted([r for r in dev if current_action(r)], key=lambda r: r["question_id"])
    # Swapped videos have a valid answer to the SAME question in the dev set.
    assert len({r["question"] for r in actions}) == 1
    assert len({r["segment_id"] for r in actions}) == len(actions) > 1
    donors = list(actions)
    random.Random(args.seed + 1).shuffle(donors)
    donor_map = {row["question_id"]: donors[(i + 1) % len(donors)]
                 for i, row in enumerate(donors)}
    correct = sorted(sample + [r for r in actions if r["question_id"] not in sampled],
                     key=lambda r: r["question_id"])
    rows = []
    for row in correct:
        row = dict(row)
        row["diagnostic"] = {"condition": "correct", "original_question_id": row["question_id"],
                             "random_dev_sample": row["question_id"] in sampled,
                             "current_action": current_action(row)}
        rows.append(row)
    for row in actions:
        donor = donor_map[row["question_id"]]
        for condition in ("rephrased", "swapped", "rephrased_swapped"):
            swap = "swapped" in condition
            new = dict(row)
            new["question_id"] = row["question_id"] + "::" + condition
            if condition.startswith("rephrased"):
                new["question"] = ACTION_QUESTION
            if swap:
                new.update(segment_id=donor["segment_id"], frames=donor["frames"],
                           answers=donor["answers"])
            new["diagnostic"] = {
                "condition": condition, "original_question_id": row["question_id"],
                "random_dev_sample": False, "current_action": True,
                "original_segment_id": row["segment_id"],
                "donor_question_id": donor["question_id"] if swap else None,
            }
            if swap:
                new["diagnostic"]["original_references"] = row["answers"]
            rows.append(new)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output / "diagnostic.jsonl", rows)
    audit = {}
    for name, split in (("train", train), ("dev", dev)):
        action_rows = [r for r in split if current_action(r)]
        audit[name] = {"records": len(split), "current_action_records": len(action_rows),
                       "questions": dict(Counter(r["question"] for r in action_rows)),
                       "reference_prefixes": dict(Counter(action_prefix(r["answers"][0])
                                                          for r in action_rows))}
        dump_jsonl(output / (name + "_action_audit.jsonl"), action_rows)
    manifest = {"seed": args.seed, "random_dev_records": len(sample),
                "action_records": len(actions), "generation_records_per_model": len(rows),
                "conditions": dict(Counter(r["diagnostic"]["condition"] for r in rows)),
                "train_dev_segment_overlap": 0,
                "dev_sha256": file_sha256(args.data_dir / "dev.jsonl"),
                "diagnostic_sha256": file_sha256(output / "diagnostic.jsonl"),
                "sampling": "Uniform 512-question dev sample plus all current-action dev questions; report separately.",
                "swap_protocol": "Seeded cyclic derangement over same-question dev videos. Score donor references; separately test agreement with original references.",
                "audit": audit}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def metrics(rows):
    if not rows:
        return {"records": 0}
    return {"records": len(rows),
            "judge_accuracy": statistics.mean(r["lingo_judge_correct"] for r in rows),
            "token_f1": statistics.mean(r["token_f1"] for r in rows),
            "mean_answer_words": statistics.mean(len(r["prediction"].split()) for r in rows),
            "stopping_prefix_count": sum(action_prefix(r["prediction"]) == "stopping" for r in rows),
            "prediction_prefixes": dict(Counter(action_prefix(r["prediction"]) for r in rows))}


def paired(old, new):
    old = {r["diagnostic"]["original_question_id"]: r for r in old}
    new = {r["diagnostic"]["original_question_id"]: r for r in new}
    assert old.keys() == new.keys()
    return {"records": len(old),
            "fixed": sum(not old[k]["lingo_judge_correct"] and new[k]["lingo_judge_correct"] for k in old),
            "regressed": sum(old[k]["lingo_judge_correct"] and not new[k]["lingo_judge_correct"] for k in old)}


def report(args):
    result = {"models": {}}
    all_rows = {}
    for name in MODELS:
        rows = load_jsonl(args.output_dir / name / "scored/scored_predictions.jsonl")
        expected = load_jsonl(args.output_dir / "diagnostic.jsonl")
        assert len(rows) == len(expected) and {r["question_id"] for r in rows} == {r["question_id"] for r in expected}
        all_rows[name] = rows
        groups = {}
        for condition in ("correct", "rephrased", "swapped", "rephrased_swapped"):
            groups[condition] = [r for r in rows if r["diagnostic"]["condition"] == condition
                                 and r["diagnostic"]["current_action"]]
        random_rows = [r for r in rows if r["diagnostic"]["random_dev_sample"]]
        item = {"random_dev": metrics(random_rows), "action": {k: metrics(v) for k, v in groups.items()},
                "rephrase_paired": paired(groups["correct"], groups["rephrased"]), "visual_controls": {}}
        for correct, swapped in (("correct", "swapped"), ("rephrased", "rephrased_swapped")):
            original = {r["diagnostic"]["original_question_id"]: r for r in groups[correct]}
            controls = groups[swapped]
            n = len(controls)
            item["visual_controls"][correct] = {
                "records": n,
                "normalized_answer_changed": sum(normalize_answer(r["prediction"]) != normalize_answer(original[r["diagnostic"]["original_question_id"]]["prediction"]) for r in controls),
                "original_reference_agreement_after_swap": statistics.mean(r["original_reference_judge_correct"] for r in controls),
                "donor_reference_accuracy_after_swap": statistics.mean(r["lingo_judge_correct"] for r in controls),
                "prefix_changed": sum(action_prefix(r["prediction"]) != action_prefix(original[r["diagnostic"]["original_question_id"]]["prediction"]) for r in controls),
                "donor_prediction_matches_correct_condition": sum(
                    normalize_answer(r["prediction"]) == normalize_answer(
                        original[r["diagnostic"]["donor_question_id"]]["prediction"])
                    for r in controls),
            }
        result["models"][name] = item
    base = [r for r in all_rows["base"] if r["diagnostic"]["random_dev_sample"]]
    for name in MODELS[1:]:
        result["models"][name]["paired_vs_base_random_dev"] = paired(
            base, [r for r in all_rows[name] if r["diagnostic"]["random_dev_sample"]])
    selected = max(MODELS[1:], key=lambda name: result["models"][name]["random_dev"]["judge_accuracy"])
    result["best_sft_on_fixed_dev_sample"] = selected
    result["limitations"] = ["512-question sample, not full dev or test benchmark.",
                             "Action subset has only 48 questions; prefix buckets are lexical heuristics.",
                             "Lingo-Judge is a text-based automatic proxy; visual sensitivity does not establish temporal reasoning.",
                             "Swaps permute same-question videos; donor correctness should match the corresponding correct-video condition."]
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    lines = ["# SFT 开发集诊断", "", "固定随机 512 题与额外动作题分开报告；不在测试集选择 checkpoint。", "",
             "| 模型 | 随机 dev Judge | 动作原问法 | 动作改写 | 原问法停车数 | 改写停车数 |", "|---|---:|---:|---:|---:|---:|"]
    for name, item in result["models"].items():
        a, b = item["action"]["correct"], item["action"]["rephrased"]
        lines.append("| %s | %.2f%% | %.2f%% | %.2f%% | %s/%s | %s/%s |" % (
            name, item["random_dev"]["judge_accuracy"] * 100, a["judge_accuracy"] * 100,
            b["judge_accuracy"] * 100, a["stopping_prefix_count"], a["records"],
            b["stopping_prefix_count"], b["records"]))
    lines += ["", "开发集样本内最佳 SFT：`%s`。详细对照和局限见 report.json。" % selected]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    action_ids = {r["diagnostic"]["original_question_id"] for r in all_rows["base"]
                  if r["diagnostic"]["current_action"]}
    cases = []
    for qid in sorted(action_ids):
        case = {"question_id": qid, "models": {}}
        for name, rows in all_rows.items():
            conditions = {r["diagnostic"]["condition"]: r for r in rows
                          if r["diagnostic"]["original_question_id"] == qid}
            case["models"][name] = conditions
        cases.append(case)
    dump_jsonl(args.output_dir / "action_cases.jsonl", cases)
    print(json.dumps(result, indent=2))


def status(args):
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    total = manifest["generation_records_per_model"]
    result = {"total_per_model": total, "models": {}, "process_liveness_verified": False}
    for name in MODELS:
        path = args.output_dir / name / "predictions.jsonl"
        count = 0
        if path.exists():
            with path.open() as handle:
                for line in handle:
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        continue  # A writer may be between write() and flush().
                    count += 1
        scored = args.output_dir / name / "scored/summary.json"
        result["models"][name] = {
            "generated": count, "total": total,
            "stage": "scored" if scored.exists() else (
                "scoring_pending_or_in_progress" if count == total else "generation_incomplete"),
            "seconds_since_prediction_update": round(time.time() - path.stat().st_mtime, 1) if path.exists() else None,
        }
    result["report_ready"] = (args.output_dir / "report.json").exists()
    print(json.dumps(result, indent=2))


def wait_report(args):
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        ready = [name for name in MODELS
                 if (args.output_dir / name / "scored/summary.json").exists()]
        print(json.dumps({"stage": "waiting_for_scoring", "ready_models": ready,
                          "expected_models": list(MODELS)}), flush=True)
        if len(ready) == len(MODELS):
            report(args)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("diagnostic scoring incomplete; inspect model services before rerunning report")
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "report", "status", "wait-report"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/processed/lingoqa_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/sft_diagnosis_v1")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    {"prepare": prepare, "report": report, "status": status,
     "wait-report": wait_report}[args.action](args)
