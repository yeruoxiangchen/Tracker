#!/usr/bin/env python3
"""Matched visual-only/question-input Q-Former SFT and dev-only checkpoint selection."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl
from scripts.report_language_lora_sft import RECIPE_KEYS, read, lines

PROTOCOL = "mpp_question_qformer_sft_comparison_v1"
LABELS = ("checkpoint-001000", "checkpoint-002500", "final")
BRANCHES = ("visual", "question")


def validate_training(args, required=False):
    from scripts.report_feedback_opd import identities
    control = read(args.control_dir / "run_config.json") or {}
    completed = read(args.control_dir / "summary.json") or {}
    expected_recipe = {"training_mode": "sft_control", "records": 40000, "epochs": 2,
                       "world_size": 4, "grad_accumulation": 4, "total_steps": 5000,
                       "checkpoint_steps": [1000, 2500], "ce_weight": 1.0}
    if (not completed.get("completed") or completed.get("optimizer_steps") != 5000 or
            any(control.get(k) != v for k, v in expected_recipe.items())):
        raise ValueError("requires the completed historical four-GPU 40k CE control")
    manifest = read(args.data_dir / "manifest.json") or {}
    train_hash = file_sha256(args.data_dir / "train.jsonl")
    if (control.get("data_sha256") != train_hash or lines(args.data_dir / "train.jsonl") != 40000 or manifest.get("records") != 40000 or
            manifest.get("split_sha256", {}).get("train") != train_hash or
            manifest.get("original_qa_preserved") is not True or manifest.get("held_out_overlap") != 0):
        raise ValueError("reuse the unchanged, split-isolated original 40k QA pool")
    parent = identities(args.initial_dir, True)
    original = {"adapter": parent["adapter_identity"], "projector": parent["projector_identity"],
                "qformer": parent["qformer_adapter_sha256"]}
    if control.get("initialization") != original:
        raise ValueError("question training and CE must start from the same full SFT parent")
    if args.train_dir.resolve() in (args.control_dir.resolve(), args.initial_dir.resolve()):
        raise ValueError("new training must use a separate output directory")
    if args.eval_dir.resolve() == args.control_eval_dir.resolve():
        raise ValueError("new reports must not replace historical CE results")
    candidate = read(args.train_dir / "run_config.json")
    if candidate:
        changed = [k for k in RECIPE_KEYS if candidate.get(k) != control.get(k)]
        question = candidate.get("question_qformer", {})
        if (changed or candidate.get("training_mode") != "sft_control" or
                candidate.get("objective") != "assistant_only_ce" or candidate.get("language_lora") or
                question.get("protocol") != "mpp_question_qformer_v1" or question.get("mode") != "question"):
            raise ValueError("question Q-Former changed the agreed CE recipe: %s" % changed)
        if required:
            final = read(args.train_dir / "summary.json") or {}
            if not final.get("completed") or final.get("optimizer_steps") != 5000:
                raise ValueError("complete all 5000 question-QFormer SFT updates before comparison")
    elif required:
        raise ValueError("question-QFormer training has not started")
    return control, candidate, parent


def preflight(args):
    print(json.dumps({"status": "checking_question_contract", "data_dir": str(args.data_dir)}), flush=True)
    from transformers import BertTokenizer
    from mpp_video_qa.data import cache_file
    from mpp_video_qa.question_qformer import text_identity
    validate_training(args)
    identity = text_identity(args.bert_dir, args.question_max_len)
    tokenizer = BertTokenizer.from_pretrained(str(args.bert_dir), do_lower_case=True, local_files_only=True)
    rows = []
    for split in ("train", "dev_action", "dev_scenery", "eval"):
        rows.extend(load_jsonl(args.data_dir / (split + ".jsonl")))
    longest = 0
    for start in range(0, len(rows), 256):
        batch = rows[start:start + 256]
        if any(len(r["frames"]) != 5 or not r["question"].strip() for r in batch):
            raise ValueError("every QA requires its original five frames and nonempty question")
        encoded = tokenizer([r["question"] for r in batch], truncation=False)
        longest = max(longest, max(map(len, encoded["input_ids"])))
    if longest > args.question_max_len:
        raise ValueError("max question length %d exceeds %d; no silent truncation" % (longest, args.question_max_len))
    segments = {r["segment_id"] for r in rows}
    missing = [s for s in segments if not cache_file(args.cache_dir, s).is_file()]
    if missing:
        raise ValueError("missing original EVA caches: %d" % len(missing))
    result = {"completed": True, "records_checked": len(rows), "segments": len(segments),
              "max_question_tokens_observed": longest, "question_qformer": identity,
              "optimizer_steps_total": 5000, "new_training_started": False}
    from scripts.generate_answers import write_json_atomic
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.eval_dir / "preflight.json", result)
    return result


def evaluation_folder(args, key, split):
    if key == "parent" or key.startswith("visual_"):
        prefix = "baseline" if key == "parent" else "sft_control_" + key.split("_", 1)[1]
        historical = args.control_eval_dir / (prefix + "_" + split)
        if (historical / "scored/summary.json").is_file():
            return historical
    return args.eval_dir / (key + "_" + split)


def model_specs(args):
    from scripts.report_feedback_opd_v2 import bundle_identity
    from scripts.report_language_lora_sft import validate_saved_language_adapter
    from mpp_video_qa.question_qformer import QUESTION_FORMAT, question_sidecar
    control, candidate, parent = validate_training(args, True)
    models, specs = {"parent": parent}, {}
    for branch, folder, config in (("visual", args.control_dir, control), ("question", args.train_dir, candidate)):
        for label, step in zip(LABELS, (1000, 2500, 5000)):
            bundle = folder / label
            metadata = read(bundle / "metadata.json") or {}
            if metadata.get("identity") != config or metadata.get("optimizer_steps") != step:
                raise ValueError("checkpoint does not belong to this SFT run: %s" % bundle)
            validate_saved_language_adapter(bundle / "adapter", "qkv")
            if branch == "question":
                visual = read(question_sidecar(bundle / "qformer_adapter.pt")) or {}
                if visual.get("format") != QUESTION_FORMAT or visual.get("question_identity") != config["question_qformer"]:
                    raise ValueError("question checkpoint metadata is incomplete or changed")
            key = branch + "_" + label
            models[key] = bundle_identity(bundle)
            specs[key] = {"branch": branch, "label": label, "step": step,
                          "bundle": str(bundle.resolve()), "metadata_sha256": file_sha256(bundle / "metadata.json")}
    return models, specs, candidate


def select_checkpoints(metrics):
    return {branch: max((branch + "_" + label for label in LABELS),
                        key=lambda key: metrics[key]["lingo_judge_accuracy"]) for branch in BRANCHES}


def build_report(args, official=False):
    from scripts.report_full_qformer_scale import load_result
    from scripts.report_answer_opsd import JUDGE_PROTOCOL, paired_metrics
    from scripts.report_preference import summarize, cluster_interval
    models, specs, config = model_specs(args)
    dev_hashes = {s: file_sha256(args.data_dir / ("dev_%s.jsonl" % s)) for s in ("action", "scenery")}
    dev = read(args.eval_dir / "report.json") if official else None
    if official:
        if (not dev or dev.get("protocol") != PROTOCOL or dev.get("models") != models or
                dev.get("checkpoint_specs") != specs or dev.get("split_sha256") != dev_hashes):
            raise ValueError("a matching development report is required before official evaluation")
        selected = select_checkpoints(dev["dev_combined"]["models"])
        if selected != dev.get("selected_checkpoints"):
            raise ValueError("development selection changed")
        keys, splits = ["parent", selected["visual"], selected["question"]], ("eval",)
    else:
        keys, splits = list(models), ("action", "scenery")
    pooled = {key: [] for key in keys}
    result = {"protocol": PROTOCOL, "created_utc": datetime.now(timezone.utc).isoformat(),
              "judge_protocol": JUDGE_PROTOCOL, "selection_split": "dev", "models": models,
              "checkpoint_specs": specs, "question_qformer": config["question_qformer"],
              "split_sha256": {}, "splits": {},
              "interpretation": "Same SFT parent, 40k original QA, five EVA frames and 5000 CE updates. "
                  "Question-input Q-Former restores BERT text modules; Qwen receives the original question. "
                  "Historical visual-only CE is reused. The repeatedly used official 500 is not a new blind test."}
    protocols = set()
    for split in splits:
        data = args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
        result["split_sha256"][split] = file_sha256(data)
        split_rows = {}
        for key in keys:
            folder = evaluation_folder(args, key, split)
            rows, protocol = load_result(folder, data, models[key])
            run = read(folder / "run_manifest.json") or {}
            expected_question = config["question_qformer"] if key.startswith("question_") else None
            if run.get("question_qformer") != expected_question:
                raise ValueError("generation did not use the expected question-input bridge: %s" % key)
            if run.get("max_new_tokens") != 96 or run.get("batch_size") != 1 or run.get("max_records") != 0:
                raise ValueError("requires the original full-split greedy96/batch1 evaluation")
            protocols.add(protocol)
            split_rows[key] = rows
            pooled[key].extend(rows)
        result["splits"][split] = {"models": {k: summarize(v) for k, v in split_rows.items()}}
    if len(protocols) != 1:
        raise ValueError("language generation protocols differ")
    metrics = {k: summarize(v) for k, v in pooled.items()}

    def pair(a, b):
        return dict(paired_metrics(pooled[a], pooled[b]), **cluster_interval(pooled[a], pooled[b]))

    if not official:
        selected = select_checkpoints(metrics)
        result["dev_combined"] = {"models": metrics, "same_step": {
            label: pair("visual_" + label, "question_" + label) for label in LABELS}}
        selected_model = max(["parent", selected["visual"], selected["question"]],
                             key=lambda key: metrics[key]["lingo_judge_accuracy"])
    else:
        selected_model = dev["selected_model"]
        result["dev_report_sha256"] = file_sha256(args.eval_dir / "report.json")
    result.update(selected_checkpoints=selected, selected_model=selected_model)
    chosen = ("parent", selected["visual"], selected["question"])
    result["selected_comparison"] = {"models": {k: metrics[k] for k in chosen},
        "paired": {"parent_vs_visual": pair(chosen[0], chosen[1]),
                   "parent_vs_question": pair(chosen[0], chosen[2]),
                   "visual_vs_question": pair(chosen[1], chosen[2])}}
    parent_correct = {r["question_id"] for r in pooled["parent"] if r["lingo_judge_correct"]}
    result["by_parent_correctness"] = {}
    for name, ids in (("parent_correct", parent_correct),
                      ("parent_wrong", {r["question_id"] for r in pooled["parent"]} - parent_correct)):
        a = [r for r in pooled[chosen[1]] if r["question_id"] in ids]
        b = [r for r in pooled[chosen[2]] if r["question_id"] in ids]
        result["by_parent_correctness"][name] = dict(paired_metrics(a, b), **cluster_interval(a, b)) if a else {"records": 0}
    result["selection_note"] = "Each arm selects from 1000/2500/5000 on combined dev Judge; ties keep earlier steps. Post-selection intervals are exploratory."
    return result


def save_report(args, result, official=False):
    from scripts.generate_answers import write_json_atomic
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    stem = "official_report" if official else "report"
    write_json_atomic(args.eval_dir / (stem + ".json"), result)
    text = ["# Question-input Q-Former SFT", "", result["interpretation"], "",
            "| Split | Model | N | Judge | Token F1 |", "|---|---|---:|---:|---:|"]
    blocks = list(result["splits"].items())
    if "dev_combined" in result:
        blocks.append(("dev_combined", result["dev_combined"]))
    for split, block in blocks:
        for key, metric in block["models"].items():
            text.append("| %s | %s | %d | %.2f%% | %.4f |" %
                        (split, key, metric["records"], 100 * metric["lingo_judge_accuracy"], metric["token_f1"]))
    for key, values in result["selected_comparison"]["paired"].items():
        score, interval = values["lingo_judge_correct"], values["net_change_95pct_interval"]
        text.append("\n%s: fixed %d / regressed %d / net %+.2fpp; segment-bootstrap 95%% CI [%+.2f, %+.2f]pp." %
                    (key, score["fixed"], score["regressed"], 100 * score["net_change_mean"],
                     100 * interval[0], 100 * interval[1]))
    text += ["", result["selection_note"], "", "Selected on dev: `%s`." % result["selected_model"]]
    (args.eval_dir / (stem + ".md")).write_text("\n".join(text) + "\n")
    return {"json": str(args.eval_dir / (stem + ".json")), "selected_checkpoints": result["selected_checkpoints"]}


def status(args):
    summary = read(args.train_dir / "summary.json")
    progress_path = args.train_dir / "progress.json"
    failures = sorted(args.train_dir.glob("failure_rank_*.json"), key=lambda p: p.stat().st_mtime)
    if summary:
        training = dict(summary, status="completed")
    elif failures and (not progress_path.exists() or failures[-1].stat().st_mtime >= progress_path.stat().st_mtime):
        training = read(failures[-1])
    else:
        training = read(progress_path) or {"status": "initializing_or_not_started", "optimizer_steps_total": 5000}
    if training.get("updated_unix_seconds"):
        training["seconds_since_update"] = round(time.time() - training["updated_unix_seconds"], 1)
    if training.get("rank0_pid") and not summary:
        try:
            os.kill(training["rank0_pid"], 0)
            training["rank0_process_alive"] = True
        except ProcessLookupError:
            training.update(rank0_process_alive=False, status="interrupted")
        except PermissionError:
            training["rank0_process_alive"] = None
    evaluation = {}
    totals = {s: lines(args.data_dir / ("eval.jsonl" if s == "eval" else "dev_%s.jsonl" % s))
              for s in ("action", "scenery", "eval")}
    for label in LABELS:
        for split in totals:
            folder = args.eval_dir / ("question_" + label + "_" + split)
            evaluation[label + "_" + split] = {"generated": lines(folder / "predictions.jsonl"),
                "total": totals[split], "scored": (folder / "scored/summary.json").is_file(),
                "score_progress": read(folder / "scored/score_progress.json")}
    return {"output_dir": str(args.train_dir), "training": training, "evaluation": evaluation,
            "dev_report_ready": (args.eval_dir / "report.json").is_file(),
            "official_report_ready": (args.eval_dir / "official_report.json").is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("data-dir", "datasets/processed/lingoqa_feedback_opd_40k_v2"),
                          ("initial-dir", "outputs/mpp_lingoqa_qformer_all_1epoch_v1"),
                          ("control-dir", "outputs/mpp_feedback_sft_control_40k_2epoch_v2"),
                          ("control-eval-dir", "outputs/feedback_opd_eval_v2"),
                          ("train-dir", "outputs/mpp_question_qformer_40k_2epoch_v1"),
                          ("eval-dir", "outputs/question_qformer_eval_v1")):
        parser.add_argument("--" + name, type=Path, default=ROOT / default)
    parser.add_argument("--bert-dir", type=Path, default=ROOT.parent / "mpp_llava_video_workspace/checkpoints/ckpt/bert-base-uncased")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1"))
    parser.add_argument("--question-max-len", type=int, default=128)
    parser.add_argument("--action", choices=("check", "status", "eval_path", "report", "report_eval", "selected"), default="status")
    parser.add_argument("--model", default="question")
    parser.add_argument("--split", choices=("action", "scenery", "eval"), default="action")
    args = parser.parse_args()
    if args.action == "status":
        value = status(args)
    elif args.action == "check":
        value = preflight(args)
    elif args.action == "eval_path":
        print(evaluation_folder(args, args.model, args.split)); return
    elif args.action == "selected":
        report = read(args.eval_dir / "report.json") or {}
        if not report or report.get("selected_checkpoints") != select_checkpoints(report["dev_combined"]["models"]):
            raise ValueError("complete development selection first")
        print(report["selected_checkpoints"][args.model]); return
    else:
        official = args.action == "report_eval"
        value = save_report(args, build_report(args, official), official)
    print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
