#!/usr/bin/env python3
"""Automatic-region input checks, progress, and matched development-only selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl
from mpp_video_qa.region_views import (REGION_VIEW_PROTOCOL, json_atomic, proposal_file, region_file,
                                      validate_plan, view_prompt)

JUDGE_PROTOCOL = "official_lower_strip_maxref_logit_gt0_maxlength128_v1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def line_count(path):
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def fingerprint(path):
    import hashlib
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError("missing/empty adapter: %s" % path)
    result = hashlib.sha256()
    for item in files:
        result.update(str(item.relative_to(path)).encode("utf-8"))
        result.update(file_sha256(item).encode("ascii"))
    return result.hexdigest()


def weight_paths(args, branch, checkpoint):
    if branch == "frozen":
        return (args.init_dir / "adapter", args.init_dir / "projector.pt", args.init_dir / "qformer_adapter.pt")
    folder = args.train_root / branch
    if checkpoint == "final":
        return (folder / "final_adapter", folder / "final_projector.pt", folder / "final_qformer_adapter.pt")
    folder = folder / checkpoint
    return (folder / "adapter", folder / "projector.pt", folder / "qformer_adapter.pt")


def summarize(rows):
    n = len(rows)
    return {"records": n, "lingo_judge_accuracy": sum(bool(r["lingo_judge_correct"]) for r in rows) / n,
            **{key: sum(float(r[key]) for r in rows) / n for key in ("normalized_exact", "token_f1")}}


def paired(base, candidate):
    candidates = {r["question_id"]: r for r in candidate}
    fixed = regressed = 0
    groups = defaultdict(list)
    for row in base:
        a, b = bool(row["lingo_judge_correct"]), bool(candidates[row["question_id"]]["lingo_judge_correct"])
        fixed += int(not a and b)
        regressed += int(a and not b)
        groups[row["segment_id"]].append(int(b) - int(a))
    values = [(sum(v), len(v)) for v in groups.values()]
    rng = random.Random(44)
    draws = []
    for _ in range(1000):
        selected = rng.choices(values, k=len(values))
        draws.append(sum(v[0] for v in selected) / sum(v[1] for v in selected))
    draws.sort()
    return {"fixed": fixed, "regressed": regressed, "net": fixed - regressed,
            "net_change_mean": (fixed - regressed) / len(base),
            "segment_bootstrap_95pct": [draws[25], draws[975]], "bootstrap_segments": len(values)}


def load_scored(args, name, split, branch, condition, checkpoint):
    folder = args.eval_dir / (name + "_" + split)
    summary = read_json(folder / "scored/summary.json")
    if summary is None:
        return None
    run = read_json(folder / "run_manifest.json") or {}
    source = args.data_dir / (split + ".jsonl")
    plan_path = args.data_dir / (split + ".views.jsonl")
    adapter, projector, bridge = weight_paths(args, branch, checkpoint)
    expected = {r["question_id"]: r for r in load_jsonl(source)}
    identity = {"data_sha256": file_sha256(source), "region_views_sha256": file_sha256(plan_path),
                "region_condition": condition, "region_view_protocol": REGION_VIEW_PROTOCOL,
                "feature_stage": "eva", "task_instruction_version": "legacy_v1",
                "max_new_tokens": 96, "batch_size": 1, "precision_protocol": "fp32_trainables_v1",
                "adapter_identity": fingerprint(adapter), "projector_identity": file_sha256(projector),
                "qformer_adapter_sha256": file_sha256(bridge)}
    if any(run.get(k) != v for k, v in identity.items()) or summary.get("judge_protocol") != JUDGE_PROTOCOL:
        raise ValueError("scored run has different input/weights/scoring: %s" % folder)
    if branch != "frozen":
        config = read_json(args.train_root / branch / "run_config.json") or {}
        init_adapter, init_projector, init_bridge = weight_paths(args, "frozen", "init")
        expected_config = {"data_sha256": file_sha256(args.data_dir / "train.jsonl"),
                           "initial_adapter_sha256": fingerprint(init_adapter),
                           "initial_projector_sha256": file_sha256(init_projector),
                           "initial_qformer_sha256": file_sha256(init_bridge),
                           "epochs": args.epochs, "feature_stage": "eva", "target_mode": "reference"}
        region = config.get("region_identity", {})
        if (any(config.get(k) != v for k, v in expected_config.items()) or
                region.get("region_condition") != condition or
                region.get("region_views_sha256") != file_sha256(args.data_dir / "train.views.jsonl") or
                region.get("dev_region_views_sha256") != file_sha256(args.data_dir / "dev.views.jsonl")):
            raise ValueError("trained branch has different initialization, QA, budget, or view plans")
    rows = load_jsonl(folder / "scored/scored_predictions.jsonl")
    if len(rows) != len(expected) or {r["question_id"] for r in rows} != set(expected):
        raise ValueError("scored run is incomplete or duplicated: %s" % folder)
    plans = {p["question_id"]: p for p in load_jsonl(plan_path)}
    for row in rows:
        target = expected[row["question_id"]]
        if (row["question"] != target["question"] or row["references"] != target["answers"] or
                row["segment_id"] != target["segment_id"] or row.get("source_subset") != target.get("source_subset")):
            raise ValueError("scored row differs from original QA")
        observation = row.get("region_observation", {})
        plan = plans[row["question_id"]]
        views = len(target["frames"]) if condition == "full" else len(plan["views"])
        if (observation.get("plan_sha256") != plan["plan_sha256"] or
                observation.get("condition") != condition or observation.get("views") != views):
            raise ValueError("prediction observation trace does not match its frozen view plan")
    if condition == "roi":
        cache_manifest = read_json(args.region_cache_dir / "cache_manifest.json") or {}
        if run.get("region_encoder_identity") != cache_manifest.get("encoder_identity"):
            raise ValueError("evaluation used a different frozen region encoder")
    return rows


def make_report(args):
    variants = [("frozen_" + c, "frozen", c, "init") for c in ("full", "duplicate", "roi")]
    variants.extend((b + "_epoch-%02d" % e, b, b, "epoch-%02d" % e)
                    for b in args.branches for e in range(1, args.epochs + 1))
    results, missing, rows_by_model = {}, [], {}
    for name, branch, condition, checkpoint in variants:
        rows = load_scored(args, name, "dev", branch, condition, checkpoint)
        if rows is None:
            missing.append(name)
            continue
        rows_by_model[name] = rows
        results[name] = {"combined": summarize(rows), "by_source": {}}
        for source in ("action", "scenery"):
            selected = [r for r in rows if r.get("source_subset") == source]
            if selected:
                results[name]["by_source"][source] = summarize(selected)
    selections = {}
    for branch in args.branches:
        names = [branch + "_epoch-%02d" % e for e in range(1, args.epochs + 1)]
        # Give each branch exactly the same number of dev checkpoint opportunities.
        if not all(n in rows_by_model for n in names):
            continue
        best = max(names, key=lambda n: results[n]["combined"]["lingo_judge_accuracy"])
        selections[branch] = {"name": best, "checkpoint": best[len(branch) + 1:],
                              "selection_split": "dev", "tie_break": "earlier_epoch"}
    if len(selections) > 1:
        configs = [read_json(args.train_root / b / "run_config.json") for b in selections]
        matched = ("data_sha256", "epochs", "total_steps", "world_size", "batch_size", "grad_accumulation",
                   "seed", "lr", "projector_lr", "qformer_lr", "qformer_mode", "precision_protocol",
                   "gradient_checkpointing", "answer_only_logits", "max_txt_len", "target_mode")
        if any(any(c.get(k) != configs[0].get(k) for k in matched) for c in configs[1:]):
            raise ValueError("SFT control and ROI training budgets/configurations are not matched")
    comparisons = {}
    compare_names = [("frozen_full", "frozen_duplicate"), ("frozen_duplicate", "frozen_roi"),
                     ("frozen_full", "frozen_roi")]
    for branch, selection in selections.items():
        compare_names.append(("frozen_full", selection["name"]))
        compare_names.append(("frozen_" + branch, selection["name"]))
    if "duplicate" in selections and "roi" in selections:
        compare_names.append((selections["duplicate"]["name"], selections["roi"]["name"]))
    if "full" in selections and "roi" in selections:
        compare_names.append((selections["full"]["name"], selections["roi"]["name"]))
    for base, candidate in compare_names:
        if base in rows_by_model and candidate in rows_by_model:
            comparisons[base + "_vs_" + candidate] = paired(rows_by_model[base], rows_by_model[candidate])
    result = {"protocol": REGION_VIEW_PROTOCOL, "judge_protocol": JUDGE_PROTOCOL,
              "development_complete": not missing, "missing_dev_runs": missing,
              "models": results, "selected": selections, "paired": comparisons,
              "official": {}, "official_paired": {}, "official_selection_was_from_dev": True,
              "caution": "Automatic detector proposals are not object-identity ground truth. Manual ROI gains are not this experiment's results. The reused official 500 is not a new blind test."}
    official_rows = {}
    if selections:
        frozen = load_scored(args, "frozen_full", "eval", "frozen", "full", "init")
        if frozen is not None:
            result["official"]["frozen_full"] = summarize(frozen)
            official_rows["frozen_full"] = frozen
        for branch, selection in selections.items():
            rows = load_scored(args, selection["name"], "eval", branch, branch, selection["checkpoint"])
            if rows is not None:
                result["official"][selection["name"]] = summarize(rows)
                official_rows[selection["name"]] = rows
                if frozen is not None:
                    result["official_paired"]["frozen_full_vs_" + selection["name"]] = paired(frozen, rows)
        for control in ("full", "duplicate"):
            if control in selections and "roi" in selections:
                a, b = selections[control]["name"], selections["roi"]["name"]
                if a in official_rows and b in official_rows:
                    result["official_paired"][a + "_vs_" + b] = paired(official_rows[a], official_rows[b])
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    json_atomic(args.eval_dir / "report.json", result)
    lines = ["# 自动问题相关局部观察：实验报告", "", "开发集选 checkpoint；官方 500 仅评测选定模型。", "",
             "| 模型 | dev Judge | Action | Scenery | Token F1 |", "|---|---:|---:|---:|---:|"]
    for name, value in results.items():
        sources = value["by_source"]
        lines.append("| %s | %.2f%% | %.2f%% | %.2f%% | %.4f |" %
                     (name, 100 * value["combined"]["lingo_judge_accuracy"],
                      100 * sources.get("action", {}).get("lingo_judge_accuracy", 0),
                      100 * sources.get("scenery", {}).get("lingo_judge_accuracy", 0), value["combined"]["token_f1"]))
    lines.extend(["", "配对修复／退化与 segment bootstrap：", "", "```json",
                  json.dumps(comparisons, ensure_ascii=False, indent=2), "```", "",
                  "仍待完成：" + (", ".join(missing) if missing else "无"), "",
                  "选模：`" + json.dumps(selections, ensure_ascii=False) + "`", "",
                  result["caution"], ""])
    if result["official"]:
        lines.extend(["## 官方 500：仅评测 dev 选定模型", "", "| 模型 | Judge | Token F1 |", "|---|---:|---:|"])
        for name, value in result["official"].items():
            lines.append("| %s | %.2f%% | %.4f |" % (name, 100 * value["lingo_judge_accuracy"], value["token_f1"]))
        lines.extend(["", "```json", json.dumps(result["official_paired"], ensure_ascii=False, indent=2), "```", ""])
    (args.eval_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def phase_status(directory):
    states = []
    names = {p.name.replace("failure_", "progress_") for p in directory.glob("failure_rank_*.json")}
    names.update(p.name for p in directory.glob("progress_rank_*.json"))
    for name in sorted(names):
        value = read_json(directory / name) or {}
        failure = read_json(directory / name.replace("progress_", "failure_"))
        if failure and failure.get("updated_unix_seconds", 0) > value.get("updated_unix_seconds", 0):
            value = failure
        if "updated_unix_seconds" in value:
            value["seconds_since_update"] = round(time.time() - value["updated_unix_seconds"], 1)
        states.append(value)
    return {"rank_progress": states, "process_liveness_verified": False}


def status(args):
    plan_manifest = read_json(args.data_dir / "manifest.json")
    result = {"proposals": phase_status(args.proposal_dir), "region_cache": phase_status(args.region_cache_dir),
              "plan_counts": ({"unique_regions": plan_manifest["unique_regions"], "splits": plan_manifest["splits"]}
                              if plan_manifest else None), "training": {}, "evaluation": {}}
    summary = read_json(args.region_cache_dir / "cache_summary.json")
    if summary:
        result["region_cache"]["summary"] = summary
    for branch in args.branches:
        folder = args.train_root / branch
        failures = sorted(folder.glob("failure_rank_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        result["training"][branch] = (read_json(failures[0]) if failures else
                                       read_json(folder / "summary.json") or read_json(folder / "progress.json") or
                                       {"status": "not_started"})
    for folder in sorted(args.eval_dir.glob("*_dev")) + sorted(args.eval_dir.glob("*_eval")):
        if folder.is_dir():
            split = "dev" if folder.name.endswith("_dev") else "eval"
            total = plan_manifest["splits"][split]["records"] if plan_manifest else None
            result["evaluation"][folder.name] = {"generated": line_count(folder / "predictions.jsonl"), "total": total,
                "scored": (folder / "scored/summary.json").is_file(),
                "score_progress": read_json(folder / "scored/score_progress.json")}
    result["report_ready"] = bool((read_json(args.eval_dir / "report.json") or {}).get("development_complete"))
    return result


def preflight(args):
    from types import SimpleNamespace
    from transformers import AutoTokenizer
    from mpp_video_qa.core import TASK_INSTRUCTION
    from mpp_video_qa.data import cache_file, CachedQADataset
    from mpp_video_qa.model import assistant_only_preprocess
    manifest = read_json(args.data_dir / "manifest.json")
    if not manifest or not manifest["qa_values_unchanged"] or manifest["split_segment_overlap"]:
        raise ValueError("original QA preservation / split isolation not established")
    cache = read_json(args.region_cache_dir / "cache_manifest.json")
    if not cache or cache["catalog_sha256"] != manifest["region_catalog_sha256"]:
        raise ValueError("selected crop cache has different catalog")
    init_paths = weight_paths(args, "frozen", "init")
    if not all(path.exists() for path in init_paths):
        raise FileNotFoundError("missing CE baseline three-part initialization bundle")
    if cache["encoder_identity"]["frozen_ln_vision_sha256_source"] != file_sha256(args.bridge_init):
        raise ValueError("original full-cache visual initialization differs from crop encoder")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False, trust_remote_code=True,
                                             local_files_only=True, padding_side="right")
    tokenizer.pad_token_id = tokenizer.eod_id
    backbone = SimpleNamespace(llm_tokenizer=tokenizer, max_txt_len=args.max_txt_len,
                               replace_image_string="<|extra_0|>",
                               replace_image_token_id=tokenizer("<|extra_0|>").input_ids[0])
    used_segments, used_regions = set(), set()
    max_sequence = max_generation = 0
    records = 0
    for split in ("train", "dev", "eval"):
        rows = load_jsonl(args.data_dir / (split + ".jsonl"))
        plans = {p["question_id"]: p for p in load_jsonl(args.data_dir / (split + ".views.jsonl"))}
        if len(plans) != len(rows):
            raise ValueError("view plans do not cover source QA")
        for row in rows:
            plan = plans[row["question_id"]]
            validate_plan(plan, row)
            used_segments.add(row["segment_id"])
            for view in plan["views"]:
                if view["kind"] == "region":
                    used_regions.add(view["region_key"])
            # ROI and duplicate have identical text lengths; check every QA.
            prompt = view_prompt(row["question"], plan, "roi", TASK_INSTRUCTION)
            encoded = assistant_only_preprocess(backbone, [prompt], [row["answers"][0]], 32)
            max_sequence = max(max_sequence, encoded["input_ids"].shape[1])
            encoded = assistant_only_preprocess(backbone, [prompt], [""], 32, for_generation=True)
            max_generation = max(max_generation, encoded["input_ids"].shape[1] + 96)
            records += 1
            if records % 1000 == 0:
                print(json.dumps({"status": "checking_token_contract", "records": records}), flush=True)
    if max_generation > args.max_txt_len:
        raise ValueError("generation prompt + 96 tokens exceeds configured context budget")
    missing_full = [s for s in used_segments if not cache_file(args.cache_dir, s).is_file()]
    missing_roi = [r for r in used_regions if not region_file(args.region_cache_dir, r).is_file()]
    if missing_full or missing_roi:
        raise ValueError("missing full/crop features: %d / %d" % (len(missing_full), len(missing_roi)))
    # Materialize a high-view observation to catch feature/prompt interface errors.
    dataset = CachedQADataset(args.data_dir / "dev.jsonl", args.cache_dir, "eva",
                              region_views=args.data_dir / "dev.views.jsonl",
                              region_cache_dir=args.region_cache_dir, region_condition="roi")
    largest = max(range(len(dataset)), key=lambda i: len(dataset.plans[dataset.rows[i]["question_id"]]["views"]))
    example = dataset[largest]
    result = {"completed": True, "records_checked": records, "max_supervised_sequence": max_sequence,
              "max_generation_prompt_plus_96": max_generation, "max_txt_len": args.max_txt_len,
              "original_segments": len(used_segments), "selected_unique_regions": len(used_regions),
              "materialized_feature_shape": list(example["features"].shape),
              "data_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
              "region_encoder_identity": cache["encoder_identity"]}
    json_atomic(args.data_dir / "preflight.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", choices=("status", "report", "preflight"), default="status")
    for name in ("data-dir", "proposal-dir", "region-cache-dir", "cache-dir", "train-root", "eval-dir", "init-dir"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--branches", nargs="+", choices=("full", "duplicate", "roi"), default=["roi"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--tokenizer", default="")
    p.add_argument("--bridge-init", default="")
    p.add_argument("--max-txt-len", type=int, default=1536)
    a = p.parse_args()
    value = status(a) if a.action == "status" else preflight(a) if a.action == "preflight" else make_report(a)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
