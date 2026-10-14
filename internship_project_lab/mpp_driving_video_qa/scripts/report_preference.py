#!/usr/bin/env python3
"""File-based progress and paired no-hint held-out comparison."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import dump_jsonl, file_sha256, load_jsonl
from scripts.report_answer_opsd import JUDGE_PROTOCOL, read_json, line_count, paired_metrics
from scripts.generate_answers import directory_fingerprint, write_json_atomic


def summarize(rows):
    n = len(rows)
    return {"records": n, "lingo_judge_accuracy": sum(r["lingo_judge_correct"] for r in rows) / n,
            "token_f1": sum(r["token_f1"] for r in rows) / n,
            "normalized_exact": sum(r["normalized_exact"] for r in rows) / n}


def cluster_interval(base, candidate, draws=1000):
    """Paired bootstrap resamples videos, not correlated questions independently."""
    c = {r["question_id"]: r for r in candidate}
    groups = defaultdict(list)
    for row in base:
        groups[row["segment_id"]].append(int(c[row["question_id"]]["lingo_judge_correct"]) - int(row["lingo_judge_correct"]))
    values = [(sum(v), len(v)) for v in groups.values()]
    rng = random.Random(42)
    boot = []
    for _ in range(draws):
        picked = rng.choices(values, k=len(values))
        boot.append(sum(x[0] for x in picked) / sum(x[1] for x in picked))
    boot.sort()
    return {"unit": "segment_id", "segments": len(values), "draws": draws,
            "net_change_95pct_interval": [boot[int(draws * .025)], boot[min(draws - 1, int(draws * .975))]]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--action", choices=("status", "report"), default="status")
    for name in ("data-dir", "reference-dir", "train-dir", "control-dir", "init-dir", "eval-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--checkpoint", default="final")
    args = p.parse_args()
    if args.action == "status":
        selection = read_json(args.data_dir / "manifest.json") or {}
        result = {"process_liveness_verified": False,
                  "selection": {k: selection.get(k) for k in ("records", "pairs", "replay", "pair_types", "visually_reviewed_pairs")},
                  "reference": read_json(args.reference_dir / "progress.json") or {"status": "not_started"},
                  "training": {}, "evaluation": {}}
        for mode, folder in (("dpo", args.train_dir), ("sft_control", args.control_dir)):
            failures = sorted(folder.glob("failure_rank_*.json"))
            value = (read_json(failures[0]) if failures else read_json(folder / "summary.json") or
                     read_json(folder / "progress.json") or {"status": "not_started"})
            if "updated_unix_seconds" in value:
                value["seconds_since_update"] = round(time.time() - value["updated_unix_seconds"], 1)
            result["training"][mode] = value
        for mode in ("joint", "sft_control", "dpo"):
            for split in ("action", "scenery", "eval"):
                folder = args.eval_dir / (mode + "_" + split)
                source = args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
                result["evaluation"][mode + "_" + split] = {
                    "generated": line_count(folder / "predictions.jsonl"), "total": line_count(source),
                    "scored": (folder / "scored/summary.json").exists(),
                    "score_progress": read_json(folder / "scored/score_progress.json")}
        result["report_ready"] = (args.eval_dir / "report.json").exists()
        print(json.dumps(result, indent=2))
        return
    # Ensure this is a matched comparison, not a different CE coefficient/data run.
    configs = [read_json(f / "run_config.json") for f in (args.train_dir, args.control_dir)]
    if not all(configs):
        raise ValueError("both training configurations are required")
    matched_keys = ("data_sha256", "reference_sha256", "reference_manifest_sha256", "initialization", "lr", "projector_lr",
                    "chosen_ce_weight", "replay_ce_weight", "epochs", "world_size", "grad_accumulation", "total_steps", "seed")
    if any(configs[0][k] != configs[1][k] for k in matched_keys) or [c["mode"] for c in configs] != ["dpo", "sft_control"]:
        raise ValueError("DPO and control are not matched")
    weights = {"joint": (args.init_dir / "final_adapter", args.init_dir / "final_projector.pt"),
               "dpo": (args.train_dir / args.checkpoint / "adapter", args.train_dir / args.checkpoint / "projector.pt"),
               "sft_control": (args.control_dir / args.checkpoint / "adapter", args.control_dir / args.checkpoint / "projector.pt")}
    identities = {k: (directory_fingerprint(a), file_sha256(v)) for k, (a, v) in weights.items()}
    result = {"judge_protocol": JUDGE_PROTOCOL, "checkpoint": args.checkpoint, "splits": {},
              "interpretation": "Preference loss is a training diagnostic, not visual QA accuracy. All responses use original full frames without answer hints."}
    pooled = {k: [] for k in weights}
    cases = []
    for split in ("action", "scenery", "eval"):
        folders = {k: args.eval_dir / (k + "_" + split) for k in weights}
        if split == "eval" and not all((f / "scored/summary.json").exists() for f in folders.values()):
            continue
        data_path = args.data_dir / ("eval.jsonl" if split == "eval" else "dev_%s.jsonl" % split)
        expected = {r["question_id"]: r for r in load_jsonl(data_path)}
        rows = {}
        protocols = []
        for mode, folder in folders.items():
            summary = read_json(folder / "scored/summary.json") or {}
            run = read_json(folder / "run_manifest.json") or {}
            if (summary.get("judge_protocol") != JUDGE_PROTOCOL or run.get("data_sha256") != file_sha256(data_path) or
                    run.get("adapter_identity") != identities[mode][0] or run.get("projector_identity") != identities[mode][1]):
                raise ValueError("missing or mismatched scoring/model/data: %s" % folder)
            protocols.append(tuple(run.get(k) for k in ("protocol", "cache_dir", "feature_stage", "task_instruction_version", "max_new_tokens", "batch_size")))
            if run.get("task_instruction_version") != "legacy_v1" or run.get("feature_stage") != "qformer":
                raise ValueError("unexpected evaluation prompt or visual stage")
            rows[mode] = load_jsonl(folder / "scored/scored_predictions.jsonl")
            if len(rows[mode]) != len(expected) or {r["question_id"] for r in rows[mode]} != set(expected):
                raise ValueError("incomplete scored split")
            for r in rows[mode]:
                target = expected[r["question_id"]]
                if r["references"] != target["answers"] or r["question"] != target["question"] or r["segment_id"] != target["segment_id"]:
                    raise ValueError("scored answers differ from evaluation data")
            if split != "eval":
                pooled[mode].extend(rows[mode])
        if len(set(protocols)) != 1:
            raise ValueError("model runs use different generation settings or cache paths")
        result["splits"][split] = {"models": {k: summarize(v) for k, v in rows.items()}}
        for a, b in (("joint", "dpo"), ("joint", "sft_control"), ("sft_control", "dpo")):
            result["splits"][split][a + "_vs_" + b] = paired_metrics(rows[a], rows[b])
        by_id = {k: {r["question_id"]: r for r in v} for k, v in rows.items()}
        for qid in expected:
            cases.append({"split": split, "question_id": qid, "segment_id": expected[qid]["segment_id"],
                          "question": expected[qid]["question"], "references": expected[qid]["answers"],
                          "models": {k: {field: v[qid][field] for field in ("prediction", "lingo_judge_correct", "lingo_judge_logit")} for k, v in by_id.items()}})
    result["dev_combined"] = {"models": {k: summarize(v) for k, v in pooled.items()}}
    for a, b in (("joint", "dpo"), ("joint", "sft_control"), ("sft_control", "dpo")):
        result["dev_combined"][a + "_vs_" + b] = dict(paired_metrics(pooled[a], pooled[b]), **cluster_interval(pooled[a], pooled[b]))
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.eval_dir / "report.json", result)
    dump_jsonl(args.eval_dir / "paired_cases.jsonl", cases)
    lines = ["# Preference training comparison", "", "No-hint five-frame free generation; matched CE weights.", "",
             "| Split | Model | N | Judge | Token F1 |", "|---|---|---:|---:|---:|"]
    for split, report in list(result["splits"].items()) + [("dev_combined", result["dev_combined"])]:
        for mode, metrics in report["models"].items():
            lines.append("| %s | %s | %d | %.2f%% | %.4f |" % (split, mode, metrics["records"], 100 * metrics["lingo_judge_accuracy"], metrics["token_f1"]))
    for comparison in ("joint_vs_dpo", "joint_vs_sft_control", "sft_control_vs_dpo"):
        report = result["dev_combined"][comparison]
        m = report["lingo_judge_correct"]
        interval = report["net_change_95pct_interval"]
        lines.append("\n%s: fixed %d, regressed %d, net %+.2f pp; video-bootstrap 95%% interval [%+.2f, %+.2f] pp." %
                     (comparison, m["fixed"], m["regressed"], m["net_change_mean"] * 100, interval[0] * 100, interval[1] * 100))
    lines.extend(["", result["interpretation"]])
    (args.eval_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
