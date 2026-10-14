#!/usr/bin/env python3
"""Train-only teacher correction and same-student-prefix probability diagnosis."""
import argparse
import base64
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl
from mpp_video_qa.feedback_opd import token_diagnostics
from mpp_video_qa.opd import SharedFrozenTeacher, completion_logits
from scripts.feedback_opd_common import (add_runtime_arguments, configure_model, runtime_identity,
                                         training_dataset, build_student, Judge, current_rollout)
from scripts.generate_answers import write_json_atomic
from scripts.train_sft import seed_all, gpu_memory


def report(rows, identity, expected):
    reasons = Counter(r["gate_reason"] for r in rows)
    attempted = [r for r in rows if r.get("teacher_correction") is not None]
    detailed = [r for r in rows if r.get("tokens")]
    tokens = [t for r in detailed for t in r["tokens"]]
    return {"completed": len(rows) == expected, "identity": identity, "draws": len(rows),
            "total_draws": expected, "questions": len({r["question_id"] for r in rows}),
            "gate_counts": dict(reasons), "eligible_draws": reasons["eligible"],
            "eligible_rate": reasons["eligible"] / max(1, len(rows)),
            "student_sampled_correct_rate": sum(r["student_judge"]["correct"] for r in rows) / max(1, len(rows)),
            "teacher_correction_attempts": len(attempted),
            "teacher_correction_success_rate": sum(r.get("eligible", False) for r in attempted) / max(1, len(attempted)),
            "token_diagnostic_draws": len(detailed),
            "plain_teacher_vs_student_max_abs_logit_difference": max(
                (r["plain_teacher_vs_student_max_abs_logit_difference"] for r in detailed), default=None),
            "feedback_decreased_sampled_token_probability_rate": sum(t["feedback_minus_plain"] < 0 for t in tokens) / max(1, len(tokens)),
            "token_jsd_mean": sum(t["jsd"] for t in tokens) / max(1, len(tokens)),
            "plain_teacher_token_jsd_mean": sum(t["plain_teacher_jsd"] for t in tokens) / max(1, len(tokens)),
            "sources": dict(Counter(r["source_subset"] for r in rows)),
            "visual_supervision_direction_verified": False,
            "interpretation": "Teacher sees training references. Judge success and log-prob changes are not proof of visual correction or student improvement."}


def write_review(destination, rows, limit=20):
    cases = sorted((r for r in rows if r.get("tokens")),
                   key=lambda r: (not r.get("eligible"), r["question_id"], r["sample_id"]))[:limit]
    parts = ['<!doctype html><meta charset="utf-8"><title>Feedback OPD training probe</title>',
             '<style>body{font:16px sans-serif;margin:24px;max-width:1500px}img{width:19%;height:auto}article{border-top:2px solid #bbb;padding:20px 0}pre{white-space:pre-wrap}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:5px}.scroll{overflow:auto}</style>',
             '<h1>训练集反馈 OPD 诊断</h1><p>教师看过参考答案。Judge 标签与概率变化不等于经过核验的视觉纠错；本页面不修改训练标签。</p>']
    for row in cases:
        parts.append('<article><h2>%s / draw %d / %s</h2>' %
                     (html.escape(str(row["question_id"])), row["sample_id"], html.escape(row["gate_reason"])))
        for path in row["frames"]:
            file = Path(path)
            kind = "image/png" if file.suffix.lower() == ".png" else "image/jpeg"
            encoded = base64.b64encode(file.read_bytes()).decode("ascii")
            parts.append('<img loading="lazy" src="data:%s;base64,%s">' % (kind, encoded))
        for label, value in (("问题", row["question"]), ("训练参考", row["references"]),
                             ("学生回答", row["student_answer"]), ("教师纠正", row.get("teacher_correction")),
                             ("Judge（学生／教师）", [row["student_judge"], row.get("teacher_judge")])):
            parts.append('<p><b>%s</b></p><pre>%s</pre>' %
                         (label, html.escape(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))))
        parts.append('<div class="scroll"><table><tr><th>位置</th><th>token</th><th>学生 logP</th><th>无反馈教师 logP</th><th>反馈教师 logP</th><th>变化</th><th>JSD</th><th>教师高概率候选</th></tr>')
        for token in row["tokens"]:
            candidates = ', '.join('%s:%.3f' % (repr(t["token"]), t["log_prob"]) for t in token["feedback_teacher_top"])
            parts.append('<tr><td>%d</td><td>%s</td><td>%.4f</td><td>%.4f</td><td>%.4f</td><td>%+.4f</td><td>%.5f</td><td>%s</td></tr>' %
                         (token["position"], html.escape(repr(token["token"])), token["student_log_prob"],
                          token["plain_teacher_log_prob"], token["feedback_teacher_log_prob"],
                          token["feedback_minus_plain"], token["jsd"], html.escape(candidates)))
        parts.append('</table></div></article>')
    destination.write_text('\n'.join(parts), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    add_runtime_arguments(p)
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--token-cases", type=int, default=40)
    p.add_argument("--review-cases", type=int, default=20)
    args = configure_model(p.parse_args(), train=False)
    if args.num_samples < 1 or args.token_cases < 1:
        raise ValueError("positive draws and diagnostic-case limit required")
    return args


def main(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(runtime_identity(args), num_samples=args.num_samples, token_cases=args.token_cases)
    path = output / "run_config.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("probe configuration changed; use a new output directory")
    write_json_atomic(path, identity)
    dataset = training_dataset(args)
    expected = {(r["question_id"], draw) for r in dataset.rows for draw in range(args.num_samples)}
    result_path = output / "predictions.jsonl"
    rows = load_jsonl(result_path) if result_path.exists() else []
    done = {(r["question_id"], r["sample_id"]) for r in rows}
    originals = {r["question_id"]: r for r in dataset.rows}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError("existing probe rows are duplicated or outside this experiment")
    for row in rows:
        original = originals[row["question_id"]]
        if any(row[key] != original[source] for key, source in
               (("segment_id", "segment_id"), ("question", "question"), ("references", "answers"), ("frames", "frames"))):
            raise ValueError("probe metadata changed")
    if done != expected:
        torch.set_num_threads(args.cpu_threads)
        device = torch.device("cuda", 0)
        write_json_atomic(output / "progress.json", {"status": "loading_model", "completed": len(rows),
                                                     "total": len(expected), "rank0_pid": os.getpid(),
                                                     "updated_unix_seconds": time.time()})
        print(json.dumps({"status": "loading_model", "initial_dir": args.initial_dir}), flush=True)
        student = build_student(args, device, train=False)
        teacher = SharedFrozenTeacher(student, args)
        teacher.validate_initial_copy()
        print(json.dumps({"status": "loading_judge", "device": args.judge_device}), flush=True)
        judge = Judge(args)
        detailed = sum(bool(r.get("tokens")) for r in rows)
        started = time.time()
        starting_draws = len(rows)
        with result_path.open("a", encoding="utf-8") as stream, torch.no_grad():
            for index, original in enumerate(dataset.rows):
                pending = [draw for draw in range(args.num_samples) if (original["question_id"], draw) not in done]
                if not pending:
                    continue
                record = dataset[index]
                features = record["features"].unsqueeze(0).to(device)
                for draw in pending:
                    draw_seed = int(hashlib.sha256(('%d:%s:%d' % (args.seed, original["question_id"], draw)).encode()).hexdigest()[:8], 16)
                    seed_all(draw_seed)
                    ids, prompt, diagnostic = current_rollout(student, teacher, judge, features, original,
                                                             args.max_new_tokens, args.teacher_mode)
                    row = dict(diagnostic, question_id=original["question_id"], segment_id=original["segment_id"],
                               question=original["question"], references=original["answers"], frames=original["frames"],
                               source_subset=original["source_subset"], sample_id=draw, draw_seed=draw_seed)
                    if prompt is not None and len(ids) and detailed < args.token_cases:
                        try:
                            student_logits = completion_logits(student, features, original["question"], ids)
                            with teacher.active() as frozen:
                                plain_logits = completion_logits(frozen, features, original["question"], ids)
                                feedback_logits = completion_logits(frozen, features, original["question"], ids, prompt)
                            difference = float((student_logits.float() - plain_logits.float()).abs().max())
                            row["plain_teacher_vs_student_max_abs_logit_difference"] = difference
                            row["tokens"] = token_diagnostics(student_logits, plain_logits, feedback_logits,
                                                               ids, student.backbone.llm_tokenizer)
                            detailed += 1
                            del student_logits, plain_logits, feedback_logits
                        except ValueError as error:
                            if "exceeds max_txt_len" not in str(error):
                                raise
                            row.update(gate_reason="teacher_context_too_long", eligible=False, context_error=str(error))
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                    rows.append(row)
                    done.add((original["question_id"], draw))
                    progress = {"status": "probing", "completed": len(rows), "total": len(expected),
                                "rank0_pid": os.getpid(),
                                "token_diagnostic_draws": detailed, "elapsed_seconds": time.time() - started,
                                "estimated_remaining_hours": (len(expected) - len(rows)) *
                                    (time.time() - started) / max(1, len(rows) - starting_draws) / 3600,
                                "updated_unix_seconds": time.time()}
                    write_json_atomic(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        write_json_atomic(output / "memory.json", gpu_memory(device))
    result = report(rows, identity, len(expected))
    write_json_atomic(output / "probe_report.json", result)
    write_review(output / "review.html", rows, args.review_cases)
    text = ['# Feedback OPD training probe', '', 'Teacher reference access is training-only. No student improvement is measured here.', '',
            '```json', json.dumps(result, indent=2, ensure_ascii=False), '```', '',
            'Inspect review.html: judge acceptance and token probability movement do not establish visual factual correctness.']
    (output / "probe_report.md").write_text('\n'.join(text) + '\n', encoding="utf-8")
    write_json_atomic(output / "progress.json", {"status": "completed", "completed": len(rows),
                                                "total": len(expected), "updated_unix_seconds": time.time()})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    cli = parse_args()
    try:
        main(cli)
    except Exception:
        folder = Path(cli.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        write_json_atomic(folder / "progress.json", {"status": "failed", "rank0_pid": os.getpid(),
                          "updated_unix_seconds": time.time(), "traceback": traceback.format_exc()})
        raise
