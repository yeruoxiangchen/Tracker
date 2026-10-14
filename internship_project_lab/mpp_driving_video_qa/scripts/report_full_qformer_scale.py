#!/usr/bin/env python3
"""Compare original joint, the retained 100k checkpoint, and full-data SFT."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, file_sha256
from scripts.generate_answers import directory_fingerprint, write_json_atomic
from scripts.report_answer_opsd import JUDGE_PROTOCOL, paired_metrics, line_count
from scripts.report_preference import cluster_interval, summarize

MODEL_ORDER = ("qformer100k", "joint", "full")  # Preserve current best on ties.
PROTOCOL_KEYS = ("protocol", "cache_dir", "feature_stage", "task_instruction_version",
                 "task_instruction", "precision_protocol", "dtype", "max_new_tokens",
                 "max_records", "batch_size", "bridge_init_sha256")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def model_specs(args):
    return {
        "joint": (args.reference_eval_dir, "joint", args.joint_dir, True),
        "qformer100k": (args.reference_eval_dir, "epoch-02", args.reference_train_dir / "epoch-02", False),
        "full": (args.eval_dir, "epoch-01", args.train_dir / "epoch-01", False),
    }


def identities(spec):
    _, _, weights, original = spec
    return {"adapter_identity": directory_fingerprint(weights / ("final_adapter" if original else "adapter")),
            "projector_identity": file_sha256(weights / ("final_projector.pt" if original else "projector.pt")),
            "qformer_adapter_sha256": None if original else file_sha256(weights / "qformer_adapter.pt")}


def load_result(folder, data_path, identity):
    run = read(folder / "run_manifest.json")
    expected_rows = load_jsonl(data_path)
    expected = {r["question_id"]: r for r in expected_rows}
    if len(expected) != len(expected_rows) or not expected:
        raise ValueError("empty/duplicate evaluation dataset: %s" % data_path)
    if run["data_sha256"] != file_sha256(data_path) or any(run.get(k) != v for k, v in identity.items()):
        raise ValueError("evaluation data or model identity mismatch: %s" % folder)
    if (run.get("feature_stage"), run.get("precision_protocol"), run.get("task_instruction_version")) != (
            "eva", "fp32_trainables_v1", "legacy_v1"):
        raise ValueError("unexpected evaluation protocol: %s" % folder)
    summary = read(folder / "scored/summary.json")
    if summary.get("judge_protocol") != JUDGE_PROTOCOL:
        raise ValueError("unexpected Judge protocol: %s" % folder)
    rows = load_jsonl(folder / "scored/scored_predictions.jsonl")
    predictions = load_jsonl(folder / "predictions.jsonl")
    by_id = {r["question_id"]: r for r in predictions}
    if (len(rows) != len(expected) or len(predictions) != len(expected) or set(by_id) != set(expected) or
            {r["question_id"] for r in rows} != set(expected)):
        raise ValueError("incomplete or duplicate predictions: %s" % folder)
    for row in rows:
        original = expected[row["question_id"]]
        if (row["segment_id"], row["question"], row["references"]) != (
                original["segment_id"], original["question"], original["answers"]):
            raise ValueError("scored metadata differs from source: %s" % folder)
        if any(row[k] != by_id[row["question_id"]][k] for k in
               ("segment_id", "question", "references", "prediction")):
            raise ValueError("scores belong to different generated answers: %s" % folder)
    actual = summarize(rows)
    if any(abs(summary.get(k, -1) - v) > 1e-9 for k, v in actual.items()):
        raise ValueError("summary does not match scored answers: %s" % folder)
    return rows, tuple(run[k] for k in PROTOCOL_KEYS)


def comparisons(rows):
    result = {}
    for a, b in (("joint", "qformer100k"), ("joint", "full"), ("qformer100k", "full")):
        if a in rows and b in rows:
            result[a + "_vs_" + b] = dict(paired_metrics(rows[a], rows[b]), **cluster_interval(rows[a], rows[b]))
    return result


def select_model(metrics):
    return max((m for m in MODEL_ORDER if m in metrics), key=lambda m: metrics[m]["lingo_judge_accuracy"])


def validate_training(args):
    manifest = read(args.data_dir / "manifest.json")
    train_hash = file_sha256(args.data_dir / "train.jsonl")
    if manifest["sampling"] != "all_original_training_rows" or manifest["split_sha256"]["train"] != train_hash:
        raise ValueError("full-data experiment requires the prepared all-original split")
    reference = read(args.reference_train_dir / "run_config.json")
    if manifest["records"] <= reference["records"]:
        raise ValueError("full-data experiment has not expanded training data")
    config_path = args.train_dir / "run_config.json"
    if not config_path.exists():
        return {"full_training_config_present": False, "records": manifest["records"]}
    config = read(config_path)
    same = ("initial_adapter_sha256", "initial_projector_sha256", "initial_qformer_sha256",
            "bridge_init_sha256", "qformer_mode", "precision_protocol", "feature_stage",
            "task_instruction_version", "lr", "projector_lr", "qformer_lr", "weight_decay",
            "warmup_ratio", "batch_size", "grad_accumulation", "world_size", "seed", "dtype",
            "max_txt_len", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules",
            "train_qformer", "train_projector", "gradient_checkpointing", "qformer_gradient_checkpointing")
    different = [k for k in same if config.get(k) != reference.get(k)]
    if different or config["epochs"] != 1 or config["data_sha256"] != train_hash or config["records"] != manifest["records"]:
        raise ValueError("scaling training configuration mismatch: %s" % different)
    expected_steps = math.ceil(math.ceil(math.ceil(config["records"] / config["world_size"]) /
                                         config["batch_size"]) / config["grad_accumulation"])
    if config.get("max_steps", 0) != 0 or config["total_steps"] != expected_steps:
        raise ValueError("full-data experiment must train the complete one-epoch budget")
    return {"full_training_config_present": True, "records": config["records"],
            "optimizer_steps_total": config["total_steps"], "reference_records": reference["records"],
            "reference_optimizer_steps_total": reference["total_steps"],
            "same_initialization_and_training_recipe": True}


def build_report(args, models, splits):
    specs = model_specs(args)
    hashes = {m: identities(specs[m]) for m in models}
    combined = {m: [] for m in models}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "judge_protocol": JUDGE_PROTOCOL,
              "selection_split": "dev", "models": hashes, "splits": {}}
    for split in splits:
        data = args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
        rows, protocols = {}, []
        for model in models:
            root, prefix, _, _ = specs[model]
            rows[model], protocol = load_result(root / (prefix + "_" + split), data, hashes[model])
            protocols.append(protocol)
            combined[model].extend(rows[model])
        if len(set(protocols)) != 1:
            raise ValueError("generation protocols differ across models")
        result["splits"][split] = {"models": {m: summarize(r) for m, r in rows.items()},
                                   "paired": comparisons(rows)}
    if "eval" not in splits:
        metrics = {m: summarize(r) for m, r in combined.items()}
        paired = comparisons(combined)
        result.update(dev_combined={"models": metrics, "paired": paired}, selected_model=select_model(metrics))
        result["selection_note"] = "Dev Judge point estimate; ties retain qformer100k. Official evaluation never selects weights."
        if "qformer100k_vs_full" in paired:
            result["full_gain_supported_by_video_bootstrap"] = paired["qformer100k_vs_full"]["net_change_95pct_interval"][0] > 0
    result["interpretation"] = "Scale changes both unique QA coverage and total updates; this does not isolate either causal contribution."
    return result


def save_report(args, result, name):
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.eval_dir / (name + ".json"), result)
    lines = ["# " + result.get("title", "Full Q-Former training-scale comparison"), "", result["interpretation"], "",
             "| Split | Model | N | Judge | Token F1 |", "|---|---|---:|---:|---:|"]
    blocks = list(result["splits"].items())
    if "dev_combined" in result:
        blocks.append(("dev_combined", result["dev_combined"]))
    for split, block in blocks:
        for model, metric in block["models"].items():
            lines.append("| %s | %s | %d | %.2f%% | %.4f |" %
                         (split, model, metric["records"], 100 * metric["lingo_judge_accuracy"], metric["token_f1"]))
    for split, block in blocks:
        for key, pair in block["paired"].items():
            score = pair["lingo_judge_correct"]
            ci = pair["net_change_95pct_interval"]
            lines.append("\n%s / %s: fixed %d, regressed %d, net %+.2f pp; video-bootstrap 95%% CI [%+.2f, %+.2f] pp." %
                         (split, key, score["fixed"], score["regressed"], 100 * score["net_change_mean"], ci[0] * 100, ci[1] * 100))
    lines.extend(["", "Selected using development data: `%s`." % result["selected_model"]])
    if "selection_note" in result:
        lines.extend(["", result["selection_note"]])
    if "recommended_model" in result:
        lines.extend(["", "Recommended main model after paired-interval check: `%s`." % result["recommended_model"]])
    (args.eval_dir / (name + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    for name in ("data-dir", "train-dir", "eval-dir", "joint-dir", "reference-train-dir", "reference-eval-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--action", choices=("check_references", "report", "report_eval", "status"), default="report")
    args = p.parse_args()
    if args.action == "status":
        for model, (root, prefix, _, _) in model_specs(args).items():
            for split in ("action", "scenery", "eval"):
                folder = root / (prefix + "_" + split)
                source = args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
                print("%s\t%s\t%d/%d\t%s" % (model, split, line_count(folder / "predictions.jsonl"),
                    line_count(source), "scored" if (folder / "scored/summary.json").exists() else "pending"))
        print("report_ready=%s official_report_ready=%s" % ((args.eval_dir / "report.json").is_file(),
                                                         (args.eval_dir / "official_report.json").is_file()))
        return
    training = validate_training(args)
    if args.action == "check_references":
        report = build_report(args, ("joint", "qformer100k"), ("action", "scenery"))
        print(json.dumps({"references_verified": True, "training": training, "dev": report["dev_combined"]["models"]}, indent=2))
        return
    if not training["full_training_config_present"]:
        raise ValueError("full-data training has not completed")
    summary = read(args.train_dir / "summary.json")
    if not summary.get("completed") or summary.get("optimizer_steps") != training["optimizer_steps_total"]:
        raise ValueError("full-data training has not completed its full update budget")
    result = build_report(args, ("joint", "qformer100k", "full"), ("action", "scenery"))
    result["training"] = training
    if args.action == "report":
        save_report(args, result, "report")
    else:
        selected = result["selected_model"]
        models = ("joint", "qformer100k", "full") if selected == "full" else ("joint", "qformer100k")
        official = build_report(args, models, ("eval",))
        official.update(selected_model=selected, training=training,
                        dev_selection=result["dev_combined"], full_evaluated=selected == "full")
        save_report(args, official, "official_report")


if __name__ == "__main__":
    main()
