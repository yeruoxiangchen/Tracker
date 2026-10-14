#!/usr/bin/env python3
"""Compare the fixed 48-video diagnostic under the qformer cache path."""
import argparse
import base64
from collections import Counter
import html
import json
import mimetypes
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, dump_jsonl, file_sha256

CONDITIONS = ("A_original", "B_paraphrase", "C_format")
MODELS = ("base", "old_lora", "joint", "targeted")
PARAPHRASE = "What are you doing at this moment, and what is the reason?"


def original_dir(model):
    return ROOT / "outputs/projector_sft_eval" / (model + "_action")


def prepare(output):
    cases = load_jsonl(ROOT / "outputs/full_v2_action_diagnosis_v1/cases.jsonl")
    if len(cases) != 48 or len({r["segment_id"] for r in cases}) != 48:
        raise ValueError("expected the fixed 48 dev videos")
    dev = {r["question_id"]: r for r in load_jsonl(ROOT / "datasets/processed/lingoqa_full_v2/dev_action.jsonl")}
    rows = []
    for case in cases:
        for field in ("question", "answers", "frames", "segment_id"):
            if case[field] != dev[case["question_id"]][field]:
                raise ValueError("historical diagnostic no longer matches dev")
        for condition, question in zip(CONDITIONS[1:], (
                PARAPHRASE, PARAPHRASE + ' Answer in the form "action, justification".')):
            rows.append(dict(case, question_id=case["question_id"] + "::" + condition,
                             question=question, diagnostic={"condition": condition,
                             "original_question_id": case["question_id"]}))
    output.mkdir(parents=True, exist_ok=True)
    for filename, values in (("cases.jsonl", cases), ("diagnostic.jsonl", rows)):
        path = output / filename
        if path.exists() and load_jsonl(path) != values:
            raise ValueError("existing diagnostic differs: %s" % path)
        dump_jsonl(path, values)
    print(json.dumps({"cases": 48, "new_generations_per_model": len(rows),
                      "output": str(output), "scope": "dev only; no training labels"}))


def load_group(output, model):
    orig = original_dir(model)
    added = output / model
    a = json.loads((orig / "run_manifest.json").read_text())
    b = json.loads((added / "run_manifest.json").read_text())
    for key in ("adapter_identity", "projector_identity", "cache_dir", "feature_stage",
                "task_instruction_version", "max_new_tokens", "batch_size"):
        if a.get(key) != b.get(key):
            raise ValueError("%s manifest mismatch: %s" % (model, key))
    if a["feature_stage"] != "qformer":
        raise ValueError("diagnosis requires qformer cache")
    if a["data_sha256"] != file_sha256(ROOT / "datasets/processed/lingoqa_full_v2/dev_action.jsonl"):
        raise ValueError("original dev fingerprint mismatch")
    if b["data_sha256"] != file_sha256(output / "diagnostic.jsonl"):
        raise ValueError("diagnostic fingerprint mismatch")
    tables = []
    for d in (orig, added):
        rows = load_jsonl(d / "scored/scored_predictions.jsonl")
        summary = json.loads((d / "scored/summary.json").read_text())
        generation = json.loads((d / "generation_summary.json").read_text())
        if not len(rows) == generation["total"] == generation["completed"] == summary["records"]:
            raise ValueError("incomplete scored run: %s" % d)
        if summary["judge_protocol"] != "official_lower_strip_maxref_logit_gt0_maxlength128_v1":
            raise ValueError("judge protocol differs")
        keyed = {r["question_id"]: r for r in rows}
        if len(keyed) != len(rows):
            raise ValueError("duplicate predictions")
        tables.append(keyed)
    expected = {r["question_id"]: r for r in load_jsonl(output / "diagnostic.jsonl")}
    if set(tables[1]) != set(expected):
        raise ValueError("diagnostic questions differ")
    groups = {c: {} for c in CONDITIONS}
    for case in load_jsonl(output / "cases.jsonl"):
        qid = case["question_id"]
        for c in CONDITIONS:
            key = qid if c == CONDITIONS[0] else qid + "::" + c
            row = tables[0 if c == CONDITIONS[0] else 1][key]
            question = case["question"] if c == CONDITIONS[0] else expected[key]["question"]
            if (row["question"] != question or row["references"] != case["answers"] or
                    row["segment_id"] != case["segment_id"]):
                raise ValueError("prediction metadata mismatch: %s" % key)
            groups[c][qid] = row
    return groups


def paired(a, b):
    if set(a) != set(b):
        raise ValueError("paired question sets differ")
    return {"records": len(a),
            "fixed": sum(not a[i]["lingo_judge_correct"] and b[i]["lingo_judge_correct"] for i in a),
            "regressed": sum(a[i]["lingo_judge_correct"] and not b[i]["lingo_judge_correct"] for i in a),
            "answer_changed": sum(a[i]["prediction"] != b[i]["prediction"] for i in a)}


def report(output, models):
    results = {m: load_group(output, m) for m in models}
    payload = {"models": {}, "between_models": {},
               "limitations": "48 dev videos; Judge is a text metric. New answers have not been visually adjudicated."}
    for m, groups in results.items():
        payload["models"][m] = {"conditions": {
            c: {"records": len(rows), "correct": sum(r["lingo_judge_correct"] for r in rows.values()),
                "judge_accuracy": statistics.mean(r["lingo_judge_correct"] for r in rows.values()),
                "token_f1": statistics.mean(r["token_f1"] for r in rows.values())}
            for c, rows in groups.items()},
            "wording": paired(groups[CONDITIONS[0]], groups[CONDITIONS[1]]),
            "format": paired(groups[CONDITIONS[1]], groups[CONDITIONS[2]])}
    for a, b in (("base", "joint"), ("old_lora", "joint"), ("joint", "targeted")):
        if a in results and b in results:
            payload["between_models"][a + "_vs_" + b] = {
                c: paired(results[a][c], results[b][c]) for c in CONDITIONS}
    (output / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 同路径动作诊断", "", "| 模型 | 原问 | 改写 | 加格式 |", "|---|---:|---:|---:|"]
    for m in models:
        lines.append("| " + m + " | " + " | ".join(
            "%.2f%%" % (100 * payload["models"][m]["conditions"][c]["judge_accuracy"])
            for c in CONDITIONS) + " |")
    lines += ["", payload["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    dump_jsonl(output / "case_predictions.jsonl", [dict(case, predictions={
        m: {c: results[m][c][case["question_id"]] for c in CONDITIONS} for m in models})
        for case in load_jsonl(output / "cases.jsonl")])
    print(json.dumps(payload, indent=2))


def review(output):
    cases = load_jsonl(output / "case_predictions.jsonl")
    parts = ['<!doctype html><meta charset="utf-8"><title>动作诊断</title>',
             '<style>body{font-family:system-ui;margin:24px}article{border-bottom:2px solid #aaa;padding:16px} img{width:19%;height:auto}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse;width:100%}</style>',
             '<h1>固定开发集动作诊断</h1><p>五帧从左至右按时间排列。Judge 是自动文本评分；回答的视觉正确性尚待复核。</p>']
    for case in cases:
        parts.append('<article><h2>Case %s</h2>' % case["case_number"])
        for frame in case["frames"]:
            mime = mimetypes.guess_type(frame)[0] or "image/jpeg"
            uri = "data:" + mime + ";base64," + base64.b64encode(Path(frame).read_bytes()).decode()
            parts.append('<img src="%s">' % uri)
        parts.append('<p>参考：%s</p><table><tr><th>模型</th><th>条件</th><th>回答</th><th>Judge</th></tr>' % html.escape(" / ".join(case["answers"])))
        for model, conditions in case["predictions"].items():
            for c, row in conditions.items():
                parts.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
                    html.escape(model), c, html.escape(row["prediction"]), row["lingo_judge_correct"]))
        parts.append('</table></article>')
    (output / "review.html").write_text("\n".join(parts), encoding="utf-8")
    print(output / "review.html")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "report", "review"))
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/action_targeted_diagnosis_v1")
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS[:3]))
    a = p.parse_args()
    if a.action == "prepare": prepare(a.output_dir)
    elif a.action == "report": report(a.output_dir, a.models)
    else: review(a.output_dir)


if __name__ == "__main__":
    main()
