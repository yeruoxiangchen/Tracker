#!/usr/bin/env python3
"""Matched teacher prompt/video comparison on the existing development probe."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, dump_jsonl, file_sha256, TASK_INSTRUCTIONS
from scripts.generate_vlm_teacher import write_json, model_identity
from scripts.teacher_distillation import read, run, score, checked_scored, immutable_json

VARIANTS = {"multiimage_complete": "multi_image", "video_complete": "video"}
INSTRUCTION = "driving_complete_v2"


def source_results(args):
    data = args.source_run / "data/dev_probe.jsonl"
    originals = load_jsonl(data)
    rows, identities = {}, {}
    for role in ("student", "teacher"):
        name = "student" if role == "student" else "legacy_teacher"
        folder = args.source_run / "probe_dev" / role
        rows[name] = checked_scored(folder, originals)
        identities[name] = read(folder / "run_manifest.json")
        if identities[name]["data_sha256"] != file_sha256(data):
            raise ValueError("source prediction/data identity mismatch")
    legacy = identities["legacy_teacher"]
    if (legacy["model"] != model_identity(args.teacher_model) or
            legacy["task_instruction_version"] != "legacy_v1" or
            legacy.get("input_mode", "multi_image") != "multi_image" or
            legacy.get("references_in_prompt") is not False):
        raise ValueError("A must be the same teacher's legacy multi-image experiment")
    return data, originals, rows, identities


def prepare(args):
    data, originals, rows, identities = source_results(args)
    legacy = identities["legacy_teacher"]
    config = {"protocol": "teacher_adaptation_v1", "source_run": str(args.source_run),
              "data_sha256": file_sha256(data), "records": len(originals),
              "teacher_model": identities["legacy_teacher"]["model"],
              "source_identities": identities,
              "source_scored_sha256": {role: file_sha256(args.source_run / "probe_dev" / role /
                                                       "scored/scored_predictions.jsonl") for role in ("student", "teacher")},
              "task_instruction_version": INSTRUCTION, "task_instruction": TASK_INSTRUCTIONS[INSTRUCTION],
              "min_pixels": legacy["min_pixels"], "max_pixels": legacy["max_pixels"],
              "max_new_tokens": legacy["max_new_tokens"], "decoding": legacy["decoding"],
              "frame_interval_seconds": args.frame_interval_seconds, "variants": VARIANTS}
    if config["decoding"] != "greedy":
        raise ValueError("source experiment is not greedy")
    immutable_json(args.output_dir / "run_config.json", config)
    return config, data, originals, rows, identities


def generate_variant(args, variant, config, data):
    folder = args.output_dir / variant
    run(args, args.teacher_python, "scripts/generate_vlm_teacher.py", "--data", data,
        "--model", args.teacher_model, "--output-dir", folder,
        "--input-mode", VARIANTS[variant], "--task-instruction-version", INSTRUCTION,
        "--min-pixels", config["min_pixels"], "--max-pixels", config["max_pixels"],
        "--max-new-tokens", config["max_new_tokens"], "--frame-interval-seconds", args.frame_interval_seconds)
    args.current_data = data
    score(args, folder)


def checked_variant(args, name, config, data, originals):
    folder = args.output_dir / name
    manifest = read(folder / "run_manifest.json")
    expected = {"model": config["teacher_model"], "data_sha256": file_sha256(data),
                "input_mode": VARIANTS[name], "task_instruction_version": INSTRUCTION,
                "task_instruction": config["task_instruction"], "references_in_prompt": False,
                "decoding": "greedy", "max_records": 0}
    expected.update({key: config[key] for key in ("min_pixels", "max_pixels", "max_new_tokens")})
    if name == "video_complete":
        expected.update(frame_interval_seconds=config["frame_interval_seconds"], do_sample_frames=False,
                        temporal_padding="repeat_last_before_resize")
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("%s settings differ from the comparison" % name)
    legacy_version = config["source_identities"]["legacy_teacher"]["transformers_version"]
    if manifest["transformers_version"] != legacy_version:
        raise ValueError("teacher Transformers version changed between A/B/C")
    return checked_scored(folder, originals), manifest


def answer_diagnostics(rows):
    lengths = [len(row["prediction"].split()) for row in rows]
    observations = [r["observation"] for r in rows if "observation" in r]
    latency = [r["latency_seconds"] for r in rows if "latency_seconds" in r]
    finished = [r["finished"] for r in rows if "finished" in r]
    return {"mean_words": statistics.mean(lengths), "le3_words": sum(n <= 3 for n in lengths),
            "empty_answers": sum(not r["prediction"].strip() for r in rows),
            "finished_rate": statistics.mean(finished) if len(finished) == len(rows) else None,
            "visual_tokens_mean": statistics.mean(x["visual_tokens"] for x in observations) if observations else None,
            "latency_seconds_mean": statistics.mean(latency) if latency else None}


def paired(base, candidate):
    from scripts.report_answer_opsd import paired_metrics
    from scripts.report_preference import cluster_interval
    return dict(paired_metrics(base, candidate), **cluster_interval(base, candidate))


def review_cases(args, originals, rows):
    """Bounded, interleaved fixes/regressions; no claim of automatic visual review."""
    lookups = {k: {r["question_id"]: r for r in v} for k, v in rows.items()}
    buckets = []
    for name in VARIANTS:
        for fixed in (True, False):
            bucket = []
            for row in originals:
                key = row["question_id"]
                a, b = lookups["student"][key], lookups[name][key]
                matches = (not a["lingo_judge_correct"] and b["lingo_judge_correct"]) if fixed else (
                    a["lingo_judge_correct"] and not b["lingo_judge_correct"])
                if matches:
                    bucket.append(dict(row, diagnostic_group=name + ("_fixed" if fixed else "_regressed")))
            buckets.append(bucket)
    selected, seen = [], set()
    for index in range(max((len(b) for b in buckets), default=0)):
        for bucket in buckets:
            if index < len(bucket) and bucket[index]["question_id"] not in seen:
                selected.append(bucket[index])
                seen.add(bucket[index]["question_id"])
                if len(selected) >= args.review_limit:
                    break
        if len(selected) >= args.review_limit:
            break
    if not selected or args.review_limit <= 0:
        return {"records": 0, "visually_reviewed": False}
    output = args.output_dir / "review"
    dump_jsonl(output / "cases.jsonl", selected)
    scored = []
    for name, lookup in lookups.items():
        path = output / (name + ".jsonl")
        dump_jsonl(path, [lookup[r["question_id"]] for r in selected])
        scored.append(name + "=" + str(path))
    from scripts.compare_teachers import report
    report(SimpleNamespace(data=output / "cases.jsonl", output_dir=output, baseline="student", scored=scored))
    return {"records": len(selected), "html": str(output / "review.html"), "visually_reviewed": False}


def build_report(args):
    from scripts.report_preference import summarize
    from scripts.report_answer_opsd import JUDGE_PROTOCOL
    config, data, originals, rows, identities = prepare(args)
    for name in VARIANTS:
        rows[name], identities[name] = checked_variant(args, name, config, data, originals)
    ids_by_group = {"all": {r["question_id"] for r in originals}}
    for source in sorted({r["source_subset"] for r in originals}):
        ids_by_group[source] = {r["question_id"] for r in originals if r["source_subset"] == source}
    ids_by_group["reason_question_keywords"] = {r["question_id"] for r in originals if
        any(word in r["question"].lower() for word in ("why", "reason", "justification"))}
    pairs = [("legacy_teacher", "multiimage_complete"), ("multiimage_complete", "video_complete"),
             ("student", "multiimage_complete"), ("student", "video_complete")]
    report = {"protocol": config["protocol"], "judge_protocol": JUDGE_PROTOCOL,
              "records": len(originals), "segments": len({r["segment_id"] for r in originals}),
              "data_sha256": config["data_sha256"], "groups": {}, "identities": identities,
              "diagnostics": {n: answer_diagnostics(v) for n, v in rows.items()},
              "teacher_assessment": {},
              "notes": ["Development comparison, not official benchmark accuracy.",
                        "Reason-question group is a keyword diagnostic, not human-labelled categories.",
                        "Video uses the same five source frames with repeated-last temporal padding. Spatial budgets match; temporal compression/token counts differ.",
                        "Fixes are Judge-based candidates for visual review, not verified training labels. No training is launched."]}
    for group, ids in ids_by_group.items():
        if not ids:
            continue
        selected = {name: [r for r in v if r["question_id"] in ids] for name, v in rows.items()}
        report["groups"][group] = {"models": {n: summarize(v) for n, v in selected.items()},
            "paired": {a + "_vs_" + b: paired(selected[a], selected[b]) for a, b in pairs}}
    for name in VARIANTS:
        usable = [dict(r, lingo_judge_correct=bool(r["lingo_judge_correct"] and r.get("finished") is True)) for r in rows[name]]
        comparison = paired(rows["student"], usable)
        report["teacher_assessment"][name] = {
            "usable_paired": comparison,
            "overall_advantage_detected": comparison["net_change_95pct_interval"][0] > 0,
            "potential_corrections": comparison["lingo_judge_correct"]["fixed"],
            "corrections_visually_verified": False,
            "decision": "review_overall_and_group_results_before_train_side_supervision"}
    report["review"] = review_cases(args, originals, rows)
    report["artifact_sha256"] = {name: file_sha256((args.source_run / "probe_dev" /
        ("student" if name == "student" else "teacher") if name in ("student", "legacy_teacher") else args.output_dir / name) /
        "scored/scored_predictions.jsonl") for name in rows}
    write_json(args.output_dir / "report.json", report)
    lines = ["# 教师任务适配对照", "", "| 分组 | 模型 | 题数 | Judge | Token F1 |", "|---|---|---:|---:|---:|"]
    for group, item in report["groups"].items():
        for name, metrics in item["models"].items():
            lines.append("| %s | %s | %d | %.2f%% | %.4f |" % (group, name, metrics["records"],
                         metrics["lingo_judge_accuracy"] * 100, metrics["token_f1"]))
    lines.extend(["", "## 全体逐题比较", ""])
    for name, comparison in report["groups"]["all"]["paired"].items():
        metric = comparison["lingo_judge_correct"]
        interval = comparison["net_change_95pct_interval"]
        lines.append("- %s：修复%d，退化%d，净%+.2fpp；视频bootstrap95%%区间[%+.2f, %+.2f]pp。" %
                     (name, metric["fixed"], metric["regressed"], metric["net_change_mean"] * 100,
                      interval[0] * 100, interval[1] * 100))
    lines.extend(["", "## 输入与回答诊断", "", "```json", json.dumps(report["diagnostics"], indent=2), "```", ""])
    lines.extend(report["notes"])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output_dir / "report.json"), "models": report["groups"]["all"]["models"],
                      "teacher_assessment": report["teacher_assessment"]}, indent=2), flush=True)
    return report


def status(args):
    from scripts.report_answer_opsd import line_count
    config_path = args.output_dir / "run_config.json"
    total = read(config_path)["records"] if config_path.exists() else line_count(args.source_run / "data/dev_probe.jsonl")
    workflow = args.output_dir / "workflow_progress.json"
    value = {"process_liveness_verified": False, "output_dir": str(args.output_dir), "total_per_variant": total,
             "workflow": read(workflow) if workflow.exists() else {"status": "not_started"}, "variants": {}}
    for name in VARIANTS:
        folder = args.output_dir / name
        progress = folder / "progress.json"
        scoring = folder / "scored/score_progress.json"
        summary = folder / "scored/summary.json"
        generated = line_count(folder / "predictions.jsonl")
        value["variants"][name] = {"generated": generated, "total": total,
            "stage": "scored" if summary.exists() else ("scoring_pending_or_in_progress" if generated == total and total else "generating_or_not_started"),
            "generation": read(progress) if progress.exists() else None,
            "scoring": read(scoring) if scoring.exists() else None}
        if progress.exists():
            value["variants"][name]["seconds_since_update"] = round(time.time() - progress.stat().st_mtime, 1)
    value["report_ready"] = (args.output_dir / "report.json").exists()
    print(json.dumps(value, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("compare", "multiimage", "video", "report", "status"))
    p.add_argument("--source-run", type=Path, default=ROOT / "outputs/teacher_distillation_v1")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/teacher_adaptation_v1")
    p.add_argument("--teacher-model", type=Path, default=os.environ.get("TEACHER_MODEL", str(ROOT.parent /
                   "chart_verification_vlm/checkpoints/Qwen3-VL-4B-Instruct")))
    p.add_argument("--teacher-python", default=os.environ.get("TEACHER_PYTHON", "/home/zjr/anaconda3/envs/chartvlm/bin/python"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--frame-interval-seconds", type=float, default=1.0)
    p.add_argument("--review-limit", type=int, default=40)
    args = p.parse_args()
    args.source_run = args.source_run.resolve()
    args.output_dir = args.output_dir.resolve()
    args.dry_run = False
    args.upstream = ROOT.parent / "mpp_llava_video_workspace/third_party/MPP-LLaVA"
    if args.action == "status":
        status(args)
        return
    progress = args.output_dir / "workflow_progress.json"
    write_json(progress, {"status": "running", "action": args.action, "pid": os.getpid(), "updated_unix_seconds": time.time()})
    try:
        config, data, _, _, _ = prepare(args)
        if args.action in ("compare", "multiimage"):
            generate_variant(args, "multiimage_complete", config, data)
        if args.action in ("compare", "video"):
            generate_variant(args, "video_complete", config, data)
        if args.action in ("compare", "report"):
            build_report(args)
    except Exception as error:
        write_json(progress, {"status": "failed", "action": args.action, "error": str(error), "updated_unix_seconds": time.time()})
        raise
    write_json(progress, {"status": "completed", "action": args.action, "updated_unix_seconds": time.time()})


if __name__ == "__main__":
    main()
