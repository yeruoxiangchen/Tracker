#!/usr/bin/env python3
"""Compare continuation checkpoints with the completed full-data parent, on dev only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256
from scripts.generate_answers import directory_fingerprint, write_json_atomic
from scripts.report_full_qformer_scale import load_result, read, save_report
from scripts.report_answer_opsd import JUDGE_PROTOCOL, paired_metrics, line_count
from scripts.report_preference import cluster_interval, summarize

CANDIDATES = ("checkpoint-006000", "checkpoint-012000", "checkpoint-018000", "checkpoint-024000", "final")


def identity(root, final=False):
    return {"adapter_identity": directory_fingerprint(root / ("final_adapter" if final else "adapter")),
            "projector_identity": file_sha256(root / ("final_projector.pt" if final else "projector.pt")),
            "qformer_adapter_sha256": file_sha256(root / ("final_qformer_adapter.pt" if final else "qformer_adapter.pt"))}


def same_payload(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype and
                left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same_payload(value, right[key]) for key, value in left.items())
    return type(left) is type(right) and left == right


def parent_evaluation_identity(parent):
    """torch.save archives can differ by filename while all tensors are identical."""
    import torch
    final = identity(parent, final=True)
    epoch = identity(parent / "epoch-01")
    if final["adapter_identity"] != epoch["adapter_identity"]:
        raise ValueError("parent epoch/final language adapters differ; cannot reuse baseline predictions")
    for key, name in (("projector_identity", "projector.pt"), ("qformer_adapter_sha256", "qformer_adapter.pt")):
        if final[key] != epoch[key]:
            left = torch.load(parent / ("final_" + name), map_location="cpu")
            right = torch.load(parent / "epoch-01" / name, map_location="cpu")
            threads = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                equal = same_payload(left, right)
            finally:
                torch.set_num_threads(threads)
            del left, right
            if not equal:
                raise ValueError("parent epoch/final %s tensors differ; cannot reuse baseline predictions" % name)
    return epoch


def validate_training(args, required=False):
    parent = read(args.parent_dir / "run_config.json")
    summary = read(args.parent_dir / "summary.json")
    manifest = read(args.data_dir / "manifest.json")
    train_hash = file_sha256(args.data_dir / "train.jsonl")
    if (not summary.get("completed") or summary["optimizer_steps"] != parent["total_steps"] or
        parent.get("qformer_mode") != "full" or parent.get("feature_stage") != "eva" or
        parent.get("precision_protocol") != "fp32_trainables_v1" or
        train_hash != parent["data_sha256"] or manifest["split_sha256"]["train"] != train_hash or
        manifest["sampling"] != "all_original_training_rows"):
        raise ValueError("continuation requires a completed full-QFormer parent and its full original training split")
    if args.train_dir.resolve() == args.parent_dir.resolve() or args.eval_dir.resolve() == args.parent_eval_dir.resolve():
        raise ValueError("continuation must use independent training and evaluation directories")
    expected_identity = identity(args.parent_dir, final=True)
    result = {"parent_run": str(args.parent_dir.resolve()), "parent_identity": expected_identity,
              "parent_optimizer_steps": summary["optimizer_steps"], "records": parent["records"],
              "new_optimizer_and_schedule": True, "config_present": False}
    config_path = args.train_dir / "run_config.json"
    if not config_path.is_file():
        if required:
            raise ValueError("continuation training configuration is missing")
        return result
    config = read(config_path)
    same = ("data_sha256", "records", "feature_stage", "qformer_mode", "precision_protocol", "dtype",
            "task_instruction_version", "bridge_init_sha256", "batch_size", "grad_accumulation", "world_size",
            "max_txt_len", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules", "train_qformer",
            "train_projector", "gradient_checkpointing", "qformer_gradient_checkpointing", "weight_decay")
    different = [k for k in same if config.get(k) != parent.get(k)]
    initial_keys = {"initial_adapter_sha256": "adapter_identity", "initial_projector_sha256": "projector_identity",
                    "initial_qformer_sha256": "qformer_adapter_sha256"}
    if (different or config["epochs"] != 1 or config.get("max_steps", 0) != 0 or
        config["total_steps"] != parent["total_steps"] or config.get("save_every") != 6000 or
        config.get("parent_run") != str(args.parent_dir.resolve()) or
        config.get("parent_optimizer_steps") != summary["optimizer_steps"] or
        any(config.get(k) != expected_identity[v] for k, v in initial_keys.items())):
        raise ValueError("continuation configuration or parent initialization mismatch: %s" % different)
    result.update(config_present=True, optimizer_steps_total=config["total_steps"],
                  learning_rates={k: config[k] for k in ("lr", "projector_lr", "qformer_lr")}, seed=config["seed"])
    return result


def model_identity(args, model):
    if model == "baseline":
        return parent_evaluation_identity(args.parent_dir)
    config = read(args.train_dir / "run_config.json")
    if model == "final":
        summary = read(args.train_dir / "summary.json")
        if not summary.get("completed") or summary["optimizer_steps"] != config["total_steps"]:
            raise ValueError("final continuation has not completed its full budget")
        return identity(args.train_dir, final=True)
    if model not in CANDIDATES:
        raise ValueError("unknown continuation candidate: %s" % model)
    root = args.train_dir / model
    meta = read(root / "metadata.json")
    if (meta["global_step"] != int(model.split("-")[1]) or
        meta["data_sha256"] != config["data_sha256"] or meta["total_steps"] != config["total_steps"]):
        raise ValueError("checkpoint metadata differs from continuation run")
    return identity(root)


def folder_for(args, model, split):
    if model == "baseline":
        return args.parent_eval_dir / ("epoch-01_" + split)
    return args.eval_dir / (model + "_" + split)


def compare_block(rows):
    return {"models": {m: summarize(r) for m, r in rows.items()},
            "paired": {"baseline_vs_" + m: dict(paired_metrics(rows["baseline"], r),
                                               **cluster_interval(rows["baseline"], r))
                       for m, r in rows.items() if m != "baseline"}}


def select_model(metrics):
    # Fixed order: parent wins ties, then the earliest continuation checkpoint.
    return max((m for m in ("baseline",) + CANDIDATES if m in metrics),
               key=lambda m: metrics[m]["lingo_judge_accuracy"])


def build_report(args, models=("baseline",) + CANDIDATES, splits=("action", "scenery")):
    if not models or models[0] != "baseline":
        raise ValueError("the first comparison model must be the full-data baseline")
    hashes = {m: model_identity(args, m) for m in models}
    combined = {m: [] for m in models}
    result = {"title": "Full Q-Former continuation comparison",
              "created_utc": datetime.now(timezone.utc).isoformat(), "judge_protocol": JUDGE_PROTOCOL,
              "selection_split": "dev", "models": hashes, "splits": {},
              "baseline_epoch_final_tensor_equality_verified": True,
              "interpretation": "One additional SFT stage from the full-data parent, with a fresh optimizer and lower LR. "
                                "The dataset and model architecture are retained. Bootstrap intervals are exploratory after checkpoint selection."}
    for split in splits:
        path = args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
        rows, protocols = {}, []
        for model in models:
            rows[model], protocol = load_result(folder_for(args, model, split), path, hashes[model])
            protocols.append(protocol)
            combined[model].extend(rows[model])
        if len(set(protocols)) != 1:
            raise ValueError("generation protocols differ between parent and continuation")
        result["splits"][split] = compare_block(rows)
    if "eval" not in splits:
        result["dev_combined"] = compare_block(combined)
        selected = select_model(result["dev_combined"]["models"])
        result["selected_model"] = selected
        supported = (selected != "baseline" and result["dev_combined"]["paired"]["baseline_vs_" + selected]
                     ["net_change_95pct_interval"][0] > 0)
        result["selected_gain_supported_by_video_bootstrap"] = supported
        result["recommended_model"] = selected if supported else "baseline"
        result["selection_note"] = ("Dev combined Judge point estimate selects the candidate for official evaluation; "
                                    "ties retain baseline/earlier checkpoint. Retain baseline as main model if paired CI crosses zero. "
                                    "Official results are never used to choose a checkpoint.")
    return result


def status(args, evaluation=False):
    if evaluation:
        result = {"process_liveness_verified": False, "models": {}}
        selection_path = args.eval_dir / "report.json"
        selected = read(selection_path).get("selected_model") if selection_path.is_file() else None
        for model in ("baseline",) + CANDIDATES:
            result["models"][model] = {}
            for split in ("action", "scenery", "eval"):
                folder = folder_for(args, model, split)
                data = args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
                scored = (folder / "scored/summary.json").is_file()
                stage = "scored" if scored else "pending"
                if split == "eval" and model != "baseline" and not scored:
                    stage = "awaiting_dev_selection" if selected is None else (
                        "pending" if model == selected else "not_requested")
                result["models"][model][split] = {"generated": line_count(folder / "predictions.jsonl"),
                    "total": line_count(data), "scored": scored, "stage": stage}
        result.update(report_ready=(args.eval_dir / "report.json").is_file(),
                      official_report_ready=(args.eval_dir / "official_report.json").is_file())
        print("report_ready=%s official_report_ready=%s (file status; process liveness not checked)" %
              (result["report_ready"], result["official_report_ready"]))
        for model, splits in result["models"].items():
            for split, value in splits.items():
                print("%-23s %-7s %5d/%-5d %s" %
                      (model, split, value["generated"], value["total"], value["stage"]))
        return
    else:
        files = [p for p in [args.train_dir / "summary.json", args.train_dir / "progress.json"]
                 + list(args.train_dir.glob("failure_rank_*.json")) if p.is_file()]
        latest = max(files, key=lambda p: p.stat().st_mtime) if files else None
        result = read(latest) if latest else {"status": "initializing_or_not_started"}
        if result.get("completed"):
            result["status"] = "completed"
        result.update(output_dir=str(args.train_dir.resolve()), process_liveness_verified=False)
        if latest:
            age = time.time() - result.get("updated_unix_seconds", latest.stat().st_mtime)
            result["seconds_since_update"] = round(age, 1)
            if age > 600 and result.get("status") == "running":
                result["status"] = "stale_progress_check_service"
        parent_summary = args.parent_dir / "summary.json"
        if parent_summary.is_file():
            parent_steps = read(parent_summary)["optimizer_steps"]
            result["parent_optimizer_steps"] = parent_steps
            step = result.get("optimizer_step", result.get("optimizer_steps", 0))
            result["cumulative_optimizer_steps"] = parent_steps + step
            if step and result.get("elapsed_seconds") and not result.get("completed"):
                result["estimated_remaining_hours"] = result["elapsed_seconds"] / step * max(
                    0, result.get("optimizer_steps_total", step) - step) / 3600
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    for key in ("data-dir", "train-dir", "eval-dir", "parent-dir", "parent-eval-dir"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--action", choices=("check", "status", "eval_status", "report", "report_eval"), default="report")
    args = p.parse_args()
    if args.action in ("status", "eval_status"):
        status(args, args.action == "eval_status")
        return
    training = validate_training(args, required=args.action != "check")
    if args.action == "check":
        baseline = build_report(args, models=("baseline",))
        print(json.dumps({"references_verified": True, "training": training,
                          "baseline_dev": baseline["dev_combined"]["models"]["baseline"]}, indent=2))
        return
    result = build_report(args)
    result["training"] = training
    if args.action == "report":
        save_report(args, result, "report")
    else:
        selected = result["selected_model"]
        official = build_report(args, models=("baseline",) if selected == "baseline" else ("baseline", selected), splits=("eval",))
        official.update(selected_model=selected, training=training, dev_selection=result["dev_combined"],
                        selected_gain_supported_by_video_bootstrap=result["selected_gain_supported_by_video_bootstrap"])
        save_report(args, official, "official_report")


if __name__ == "__main__":
    main()
