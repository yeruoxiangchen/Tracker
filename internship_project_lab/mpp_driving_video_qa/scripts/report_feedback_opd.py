#!/usr/bin/env python3
"""Matched no-feedback dev selection, selected official eval and file/PID progress."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256
from mpp_video_qa.feedback_opd import validate_matched_configs
from scripts.generate_answers import directory_fingerprint, write_json_atomic
from scripts.report_answer_opsd import read_json, line_count, paired_metrics, JUDGE_PROTOCOL
from scripts.report_preference import summarize, cluster_interval
from scripts.report_full_qformer_scale import load_result


def identities(folder, original=False):
    return {"adapter_identity": directory_fingerprint(folder / ("final_adapter" if original else "final/adapter")),
            "projector_identity": file_sha256(folder / ("final_projector.pt" if original else "final/projector.pt")),
            "qformer_adapter_sha256": file_sha256(folder / ("final_qformer_adapter.pt" if original else "final/qformer_adapter.pt"))}


def validate_training(args):
    configs = [read_json(path / "run_config.json") for path in (args.control_dir, args.train_dir)]
    if not all(configs):
        raise ValueError("both training runs must finish before comparison")
    validate_matched_configs(*configs)
    for folder, config in zip((args.control_dir, args.train_dir), configs):
        summary = read_json(folder / "summary.json") or {}
        if not summary.get("completed") or summary["optimizer_steps"] != config["total_steps"]:
            raise ValueError("training budget incomplete: %s" % folder)
        if config["data_sha256"] != file_sha256(args.data_dir / "train.jsonl"):
            raise ValueError("training pool differs")
    baseline = identities(args.initial_dir, True)
    expected = {"adapter": baseline["adapter_identity"], "projector": baseline["projector_identity"],
                "qformer": baseline["qformer_adapter_sha256"]}
    if configs[0]["initialization"] != expected:
        raise ValueError("evaluation baseline differs from training initialization")
    return configs


def comparisons(rows):
    pairs = (("baseline", "sft_control"), ("baseline", "feedback_opd"), ("sft_control", "feedback_opd"))
    return {a + "_vs_" + b: dict(paired_metrics(rows[a], rows[b]), **cluster_interval(rows[a], rows[b]))
            for a, b in pairs if a in rows and b in rows}


def build_report(args, official=False):
    validate_training(args)
    hashes = {"baseline": identities(args.initial_dir, True), "sft_control": identities(args.control_dir),
              "feedback_opd": identities(args.train_dir)}
    if official:
        dev = read_json(args.eval_dir / "report.json")
        if not dev or dev["models"] != hashes:
            raise ValueError("complete matching development selection is required first")
        models = tuple(dict.fromkeys(("baseline", dev["selected_model"])))
        splits = ("eval",)
    else:
        models, splits = ("baseline", "sft_control", "feedback_opd"), ("action", "scenery")
    combined = {m: [] for m in models}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "judge_protocol": JUDGE_PROTOCOL,
              "models": hashes, "selection_split": "dev", "splits": {}}
    for split in splits:
        data = args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
        rows, protocols = {}, []
        for model in models:
            rows[model], protocol = load_result(args.eval_dir / (model + "_" + split), data, hashes[model])
            protocols.append(protocol)
            combined[model].extend(rows[model])
        if len(set(protocols)) != 1:
            raise ValueError("evaluation input/generation protocols differ")
        result["splits"][split] = {"models": {m: summarize(r) for m, r in rows.items()}, "paired": comparisons(rows)}
    if official:
        result.update(selected_model=dev["selected_model"], dev_report_sha256=file_sha256(args.eval_dir / "report.json"))
    else:
        metrics = {m: summarize(r) for m, r in combined.items()}
        result["dev_combined"] = {"models": metrics, "paired": comparisons(combined)}
        result["selected_model"] = max(models, key=lambda m: metrics[m]["lingo_judge_accuracy"])
        result["selection_note"] = "Dev Judge point estimate; ties keep baseline then SFT. No official results enter selection."
        result["opd_independent_gain_supported"] = (
            metrics["feedback_opd"]["lingo_judge_accuracy"] > metrics["baseline"]["lingo_judge_accuracy"] and
            result["dev_combined"]["paired"]["sft_control_vs_feedback_opd"]["net_change_95pct_interval"][0] > 0)
    result["interpretation"] = "Teacher sees train references. Only no-feedback student evaluation versus matched CE control measures distillation benefit. Matched updates do not imply matched GPU time."
    return result


def save_report(args, result, official=False):
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    name = "official_report" if official else "report"
    write_json_atomic(args.eval_dir / (name + ".json"), result)
    lines = ["# Feedback OPD comparison", "", result["interpretation"], "",
             "| Split | Model | N | Judge | Token F1 |", "|---|---|---:|---:|---:|"]
    blocks = list(result["splits"].items())
    if "dev_combined" in result:
        blocks.append(("dev_combined", result["dev_combined"]))
    for split, block in blocks:
        for model, metric in block["models"].items():
            lines.append("| %s | %s | %d | %.2f%% | %.4f |" %
                         (split, model, metric["records"], 100 * metric["lingo_judge_accuracy"], metric["token_f1"]))
        for pair, value in block["paired"].items():
            score, interval = value["lingo_judge_correct"], value["net_change_95pct_interval"]
            lines.append("\n%s / %s: fixed %d, regressed %d, net %+.2fpp; video-bootstrap 95%% CI [%+.2f, %+.2f]pp." %
                         (split, pair, score["fixed"], score["regressed"], 100 * score["net_change_mean"],
                          100 * interval[0], 100 * interval[1]))
    lines.extend(["", "Development-selected model: `%s`." % result["selected_model"]])
    (args.eval_dir / (name + ".md")).write_text('\n'.join(lines) + '\n', encoding="utf-8")
    print(json.dumps(result, indent=2))


def status(args):
    result = {"data": read_json(args.data_dir / "manifest.json"),
              "probe": read_json(args.probe_dir / "progress.json") or {"status": "not_started"},
              "probe_report_ready": (args.probe_dir / "probe_report.json").exists(),
              "training": {}, "evaluation": {}}
    for model, folder in (("feedback_opd", args.train_dir), ("sft_control", args.control_dir)):
        failures = sorted(folder.glob("failure_rank_*.json"))
        value = (read_json(failures[0]) if failures else read_json(folder / "summary.json") or
                 read_json(folder / "progress.json") or {"status": "not_started"})
        if value.get("completed"):
            value["status"] = "completed"
        if "updated_unix_seconds" in value:
            value["seconds_since_update"] = round(time.time() - value["updated_unix_seconds"], 1)
        if "rank0_pid" in value:
            try:
                os.kill(value["rank0_pid"], 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = None
            value["rank0_process_alive"] = alive
            if not alive and value.get("status") in ("running", "loading_model"):
                value["status"] = "interrupted"
        result["training"][model] = value
    for model in ("baseline", "sft_control", "feedback_opd"):
        for split in ("action", "scenery", "eval"):
            folder = args.eval_dir / (model + "_" + split)
            data = args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
            result["evaluation"][model + "_" + split] = {
                "generated": line_count(folder / "predictions.jsonl"), "total": line_count(data),
                "scored": (folder / "scored/summary.json").exists(),
                "score_progress": read_json(folder / "scored/score_progress.json")}
    result.update(report_ready=(args.eval_dir / "report.json").exists(),
                  official_report_ready=(args.eval_dir / "official_report.json").exists())
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    for name, default in (("data-dir", "datasets/processed/lingoqa_feedback_opd_v1"),
                          ("initial-dir", "outputs/mpp_lingoqa_qformer_all_1epoch_v1"),
                          ("probe-dir", "outputs/feedback_opd_probe_v1"),
                          ("train-dir", "outputs/mpp_feedback_opd_10k_v1"),
                          ("control-dir", "outputs/mpp_feedback_sft_control_10k_v1"),
                          ("eval-dir", "outputs/feedback_opd_eval_v1")):
        p.add_argument("--" + name, type=Path, default=ROOT / default)
    p.add_argument("--action", choices=("status", "report", "report_eval", "selected"), default="status")
    args = p.parse_args()
    if args.action == "status":
        status(args)
    elif args.action == "selected":
        report = read_json(args.eval_dir / "report.json")
        if not report:
            raise ValueError("development report is required")
        print(report["selected_model"])
    else:
        official = args.action == "report_eval"
        save_report(args, build_report(args, official), official)


if __name__ == "__main__":
    main()
