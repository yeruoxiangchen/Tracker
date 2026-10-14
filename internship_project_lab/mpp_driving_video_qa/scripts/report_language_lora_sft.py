#!/usr/bin/env python3
"""Expanded language LoRA progress and matched historical-CE comparisons."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256

LABELS = ("checkpoint-001000", "checkpoint-002500", "final")
BRANCHES = ("qkv", "decoder")
RECIPE_KEYS = ("data_sha256", "initialization", "base_checkpoint_sha256", "bridge_init_sha256",
               "precision_protocol", "qformer_mode", "dtype", "task_instruction_version",
               "max_txt_len", "seed", "dropout", "records", "world_size", "epochs",
               "grad_accumulation", "total_steps", "lr", "projector_lr", "qformer_lr",
               "weight_decay", "warmup_ratio", "gradient_checkpointing", "qformer_gradient_checkpointing",
               "ce_weight", "feature_stage", "checkpoint_steps")


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def lines(path):
    if not path.is_file():
        return 0
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def validate_training(args, required=False):
    from scripts.report_feedback_opd import identities
    from mpp_video_qa.lora_expansion import scope_identity
    control = read(args.control_dir / "run_config.json") or {}
    summary = read(args.control_dir / "summary.json") or {}
    manifest = read(args.data_dir / "manifest.json") or {}
    if (not summary.get("completed") or summary.get("optimizer_steps") != 5000 or
            control.get("training_mode") != "sft_control" or control.get("records") != 40000 or
            control.get("total_steps") != 5000 or control.get("epochs") != 2 or
            control.get("world_size") != 4 or control.get("grad_accumulation") != 4 or
            control.get("checkpoint_steps") != [1000, 2500]):
        raise ValueError("the completed historical 40k / 5000-update CE control is required")
    train_hash = file_sha256(args.data_dir / "train.jsonl")
    if (control["data_sha256"] != train_hash or manifest.get("records") != 40000 or
            manifest.get("split_sha256", {}).get("train") != train_hash or
            manifest.get("original_qa_preserved") is not True or manifest.get("held_out_overlap") != 0):
        raise ValueError("reuse the unchanged original-QA 40k training pool")
    parent = identities(args.initial_dir, True)
    expected = {"adapter": parent["adapter_identity"], "projector": parent["projector_identity"],
                "qformer": parent["qformer_adapter_sha256"]}
    if control.get("initialization") != expected:
        raise ValueError("new training and historical CE must start from the same complete parent bundle")
    config = read(args.train_dir / "run_config.json")
    if config:
        changed = [key for key in RECIPE_KEYS if config.get(key) != control.get(key)]
        if (changed or config.get("training_mode") != "sft_control" or
                config.get("language_lora") != scope_identity("decoder_linear") or
                config.get("objective") != "assistant_only_ce"):
            raise ValueError("expanded LoRA changed the matched CE recipe: %s" % changed)
        if required:
            complete = read(args.train_dir / "summary.json") or {}
            if not complete.get("completed") or complete.get("optimizer_steps") != 5000:
                raise ValueError("expanded LoRA has not completed the agreed training budget")
    elif required:
        raise ValueError("expanded LoRA training has not started")
    return control, config


def model_bundle(args, key):
    if key == "parent":
        return args.initial_dir, True
    branch, label = key.split("_", 1)
    if branch not in BRANCHES or label not in LABELS:
        raise ValueError("unknown model: %s" % key)
    return (args.control_dir if branch == "qkv" else args.train_dir) / label, False


def evaluation_folder(args, key, split):
    """Reuse scored historical predictions, while writing new jobs separately."""
    prefix = "baseline" if key == "parent" else "sft_control_" + key.split("_", 1)[1]
    historical = args.control_eval_dir / (prefix + "_" + split)
    if (key == "parent" or key.startswith("qkv_")) and (historical / "scored/summary.json").is_file():
        return historical
    return args.eval_dir / (key + "_" + split)


def validate_saved_language_adapter(folder, branch, num_layers=32):
    """Validate PEFT suffix selectors AND the saved per-layer A/B tensors.

    PEFT may condense 160 full Qwen module paths to four suffixes. In
    particular, c_proj selects both attn.c_proj and mlp.c_proj. Configuration
    spelling is not a reliable measure of actual adapter coverage.
    This experiment uses the original 32-block Qwen7B; tests use small Qwen.
    Only safetensors keys/shapes are read, without loading the base model.
    """
    from safetensors import safe_open
    from mpp_video_qa.lora_expansion import DECODER_LINEAR_TARGETS

    folder = Path(folder)
    if branch not in BRANCHES or num_layers <= 0:
        raise ValueError("invalid language adapter branch or layer count")
    config = read(folder / "adapter_config.json") or {}
    if (config.get("r") != 16 or config.get("lora_alpha") != 32 or
            config.get("rank_pattern") or config.get("alpha_pattern")):
        raise ValueError("saved language adapter must use uniform rank 16 / alpha 32: %s" % folder)
    targets = config.get("target_modules")
    if (not isinstance(targets, list) or not targets or
            any(not isinstance(t, str) or not t for t in targets)):
        raise ValueError("saved language adapter requires explicit module selectors: %s" % folder)
    tails = DECODER_LINEAR_TARGETS if branch == "decoder" else ("attn.c_attn",)
    expected = {"transformer.h.%d.%s" % (layer, tail)
                for layer in range(num_layers) for tail in tails}

    def matches(name, target):
        return name == target or name.endswith("." + target)

    unmatched = [t for t in targets if not any(matches(name, t) for name in expected)]
    uncovered = sorted(name for name in expected if not any(matches(name, t) for t in targets))
    if unmatched or uncovered:
        raise ValueError("saved language adapter selectors do not cover the intended modules: %s; "
                         "unmatched=%s; uncovered=%s" % (folder, unmatched[:5], uncovered[:5]))

    weights = folder / "adapter_model.safetensors"
    if not weights.is_file():
        raise ValueError("saved language adapter weights are missing: %s" % weights)
    pattern = re.compile(r"base_model\.model\.(transformer\.h\.\d+\..+)\.lora_([AB])\.weight")
    actual, unexpected, bad_shapes = set(), [], []
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            match = pattern.fullmatch(key)
            if match is None or match.group(1) not in expected:
                unexpected.append(key)
                continue
            module, matrix = match.groups()
            actual.add((module, matrix))
            shape = handle.get_slice(key).get_shape()
            if (len(shape) != 2 or any(dim <= 0 for dim in shape) or
                    shape[0 if matrix == "A" else 1] != config["r"]):
                bad_shapes.append((key, shape))
    missing = sorted({(name, matrix) for name in expected for matrix in ("A", "B")} - actual)
    if missing or unexpected or bad_shapes:
        raise ValueError("saved language adapter tensors do not match intended coverage: %s; "
                         "missing=%s; unexpected=%s; bad_shapes=%s" %
                         (folder, missing[:5], unexpected[:5], bad_shapes[:5]))
    return {"protocol": "peft_suffix_and_saved_ab_coverage_v1", "num_layers": num_layers,
            "module_count": len(expected), "tensor_count": len(actual),
            "module_types": list(tails), "r": config["r"], "alpha": config["lora_alpha"]}


def model_specs(args):
    from scripts.report_feedback_opd import identities
    from scripts.report_feedback_opd_v2 import bundle_identity
    control, config = validate_training(args, required=True)
    hashes = {"parent": identities(args.initial_dir, True)}
    specs = {}
    for branch, run in (("qkv", control), ("decoder", config)):
        for label, step in zip(LABELS, (1000, 2500, 5000)):
            key = branch + "_" + label
            folder, _ = model_bundle(args, key)
            metadata = read(folder / "metadata.json") or {}
            if metadata.get("identity") != run or metadata.get("optimizer_steps") != step:
                raise ValueError("checkpoint metadata differs from this training run: %s" % folder)
            coverage = validate_saved_language_adapter(folder / "adapter", branch)
            hashes[key] = bundle_identity(folder)
            specs[key] = {"branch": branch, "label": label, "step": step,
                          "bundle": str(folder.resolve()), "metadata_sha256": file_sha256(folder / "metadata.json"),
                          "adapter_validation": coverage}
    return hashes, specs


def select_checkpoints(metrics):
    # Equal opportunities in both branches; stable order chooses the earlier tie.
    return {branch: max((branch + "_" + label for label in LABELS),
                        key=lambda key: metrics[key]["lingo_judge_accuracy"])
            for branch in BRANCHES}


def build_report(args, official=False):
    from scripts.report_answer_opsd import paired_metrics, JUDGE_PROTOCOL
    from scripts.report_full_qformer_scale import load_result
    from scripts.report_preference import summarize, cluster_interval
    hashes, specs = model_specs(args)
    split_hashes = {split: file_sha256(args.data_dir / ("dev_%s.jsonl" % split))
                    for split in ("action", "scenery")}
    if official:
        dev = read(args.eval_dir / "report.json") or {}
        if (dev.get("models") != hashes or dev.get("checkpoint_specs") != specs or
                dev.get("split_sha256") != split_hashes):
            raise ValueError("complete the matching development report before official evaluation")
        selected = select_checkpoints(dev["dev_combined"]["models"])
        if selected != dev.get("selected_checkpoints"):
            raise ValueError("development selection changed")
        keys, splits = ["parent", selected["qkv"], selected["decoder"]], ("eval",)
    else:
        keys, splits = list(hashes), ("action", "scenery")
    pooled = {key: [] for key in keys}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "judge_protocol": JUDGE_PROTOCOL,
              "selection_split": "dev", "models": hashes, "checkpoint_specs": specs,
              "split_sha256": {}, "splits": {},
              "interpretation": "Same full-SFT parent, original QA, five frames and 5000-update CE budget. "
                  "Only language LoRA coverage expands; rank remains 16. Historical QKV predictions are reused. "
                  "The repeatedly used official 500 is not a new blind test."}
    for split in splits:
        data = args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
        result["split_sha256"][split] = file_sha256(data)
        rows, protocols = {}, []
        for key in keys:
            rows[key], protocol = load_result(evaluation_folder(args, key, split), data, hashes[key])
            pooled[key].extend(rows[key])
            protocols.append(protocol)
        if len(set(protocols)) != 1:
            raise ValueError("generation protocols differ between historical and expanded runs")
        result["splits"][split] = {"models": {key: summarize(value) for key, value in rows.items()}}
    metrics = {key: summarize(value) for key, value in pooled.items()}
    def pair(a, b):
        return dict(paired_metrics(pooled[a], pooled[b]), **cluster_interval(pooled[a], pooled[b]))
    if not official:
        selected = select_checkpoints(metrics)
        result["dev_combined"] = {"models": metrics, "same_step_qkv_vs_decoder": {
            label: pair("qkv_" + label, "decoder_" + label) for label in LABELS}}
        selected_model = max(["parent", selected["qkv"], selected["decoder"]],
                             key=lambda key: metrics[key]["lingo_judge_accuracy"])
    else:
        selected_model = dev["selected_model"]
        result["dev_report_sha256"] = file_sha256(args.eval_dir / "report.json")
    result["selected_checkpoints"] = selected
    result["selected_model"] = selected_model
    result["selected_comparison"] = {
        "models": {key: metrics[key] for key in ("parent", selected["qkv"], selected["decoder"])},
        "paired": {"parent_vs_qkv": pair("parent", selected["qkv"]),
                   "parent_vs_decoder": pair("parent", selected["decoder"]),
                   "qkv_vs_decoder": pair(selected["qkv"], selected["decoder"])}}
    result["selection_note"] = "Select each branch from steps 1000/2500/5000 by combined dev Judge; ties choose the earlier step."
    return result


def save_report(args, result, official=False):
    from scripts.generate_answers import write_json_atomic
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    stem = "official_report" if official else "report"
    write_json_atomic(args.eval_dir / (stem + ".json"), result)
    output = ["# Expanded language LoRA SFT", "", result["interpretation"], "",
              "| Split | Model | N | Judge | Token F1 |", "|---|---|---:|---:|---:|"]
    blocks = list(result["splits"].items())
    if "dev_combined" in result:
        blocks.append(("dev_combined", result["dev_combined"]))
    for split, block in blocks:
        for key, metric in block["models"].items():
            output.append("| %s | %s | %d | %.2f%% | %.4f |" %
                          (split, key, metric["records"], 100 * metric["lingo_judge_accuracy"], metric["token_f1"]))
    pairs = list(result["selected_comparison"]["paired"].items())
    if "dev_combined" in result:
        pairs += [("same_step_" + label, value)
                  for label, value in result["dev_combined"]["same_step_qkv_vs_decoder"].items()]
    for key, value in pairs:
        score, interval = value["lingo_judge_correct"], value["net_change_95pct_interval"]
        output.append("\n%s: fixed %d / regressed %d / net %+.2fpp; video-bootstrap 95%% CI [%+.2f, %+.2f]pp." %
                      (key, score["fixed"], score["regressed"], 100 * score["net_change_mean"],
                       100 * interval[0], 100 * interval[1]))
    output += ["", result["selection_note"], "", "Development-selected model: `%s`." % result["selected_model"]]
    (args.eval_dir / (stem + ".md")).write_text("\n".join(output) + "\n")
    print(json.dumps({"report": str(args.eval_dir / (stem + ".json")),
                      "selected_model": result["selected_model"],
                      "selected_checkpoints": result["selected_checkpoints"]}, indent=2), flush=True)


def status(args):
    progress = read(args.train_dir / "progress.json") or {}
    summary = read(args.train_dir / "summary.json")
    failures = sorted(args.train_dir.glob("failure_rank_*.json"), key=lambda path: path.stat().st_mtime)
    if summary:
        training = dict(summary, status="completed")
    elif failures and (not (args.train_dir / "progress.json").exists() or
                       failures[-1].stat().st_mtime >= (args.train_dir / "progress.json").stat().st_mtime):
        training = read(failures[-1])
    else:
        training = progress or {"status": "initializing_or_not_started", "optimizer_steps_total": 5000}
    if training.get("updated_unix_seconds"):
        training["seconds_since_update"] = round(time.time() - training["updated_unix_seconds"], 1)
    if training.get("rank0_pid"):
        try:
            os.kill(training["rank0_pid"], 0)
            training["rank0_process_alive"] = True
        except ProcessLookupError:
            training["rank0_process_alive"] = False
            training["status"] = "interrupted"
        except PermissionError:
            training["rank0_process_alive"] = None
    evaluation = {}
    totals = {split: lines(args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split))
              for split in ("action", "scenery", "eval")}
    for label in LABELS:
        for split in ("action", "scenery", "eval"):
            folder = args.eval_dir / ("decoder_" + label + "_" + split)
            evaluation[label + "_" + split] = {
                "generated": lines(folder / "predictions.jsonl"),
                "total": totals[split],
                "scored": (folder / "scored/summary.json").is_file()}
    return {"output_dir": str(args.train_dir), "training": training, "evaluation": evaluation,
            "dev_report_ready": (args.eval_dir / "report.json").is_file(),
            "official_report_ready": (args.eval_dir / "official_report.json").is_file()}


def main():
    parser = argparse.ArgumentParser()
    for name, default in (("data-dir", "datasets/processed/lingoqa_feedback_opd_40k_v2"),
                          ("initial-dir", "outputs/mpp_lingoqa_qformer_all_1epoch_v1"),
                          ("control-dir", "outputs/mpp_feedback_sft_control_40k_2epoch_v2"),
                          ("control-eval-dir", "outputs/feedback_opd_eval_v2"),
                          ("train-dir", "outputs/mpp_language_lora_decoder_40k_2epoch_v1"),
                          ("eval-dir", "outputs/language_lora_eval_v1")):
        parser.add_argument("--" + name, type=Path, default=ROOT / default)
    parser.add_argument("--action", choices=("check", "status", "report", "report_eval", "selected", "eval_path"), default="status")
    parser.add_argument("--model", default="decoder_final")
    parser.add_argument("--split", choices=("action", "scenery", "eval"), default="action")
    args = parser.parse_args()
    if args.action == "status":
        print(json.dumps(status(args), indent=2))
    elif args.action == "check":
        control, config = validate_training(args)
        print(json.dumps({"completed": True, "training_config_present": bool(config),
                          "records": control["records"], "optimizer_steps_total": control["total_steps"],
                          "same_parent_and_original_qa": True, "language_scope": "decoder_linear"}, indent=2))
    elif args.action == "eval_path":
        print(evaluation_folder(args, args.model, args.split))
    elif args.action == "selected":
        report = read(args.eval_dir / "report.json") or {}
        if not report or report.get("selected_checkpoints") != select_checkpoints(report["dev_combined"]["models"]):
            raise ValueError("complete the development report first")
        key = report["selected_checkpoints"][args.model]
        folder, _ = model_bundle(args, key)
        spec = report["checkpoint_specs"][key]
        if (spec["bundle"] != str(folder.resolve()) or
                spec["metadata_sha256"] != file_sha256(folder / "metadata.json")):
            raise ValueError("selected checkpoint no longer matches the development report")
        print(key)
    else:
        official = args.action == "report_eval"
        save_report(args, build_report(args, official), official)


if __name__ == "__main__":
    main()
