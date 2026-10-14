#!/usr/bin/env python3
"""Offline evidence-search -> frozen MPP QA; no external direct-answer benchmark."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl
from mpp_video_qa.evidence_search import SEARCH_PROTOCOL, result_file, validate_result
from mpp_video_qa.region_views import EVIDENCE_VIEW_PROTOCOL, json_atomic
from scripts.prepare_evidence_search import assemble_views, export_inputs, immutable_json
from scripts.report_auto_regions import fingerprint, paired, summarize

JUDGE_PROTOCOL = "official_lower_strip_maxref_logit_gt0_maxlength128_v1"


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def split_dir(a):
    return a.data_dir / a.split


def search_dir(a):
    return a.run_dir / ("search_" + a.split)


def cache_dir(a):
    return a.cache_root / a.split


def eval_dir(a, condition):
    return a.run_dir / ("mpp_%s_%s" % (condition, a.split))


def command(a, stage, args, single_gpu=False):
    args = list(map(str, args))
    print("$ " + " ".join(shlex.quote(x) for x in args), flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PYTHONPATH"] = str(ROOT) + ":" + str(a.upstream) + ":" + env.get("PYTHONPATH", "")
    if single_gpu and env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"].split(",")[0]
    begun = time.monotonic()
    progress = {"status": "running", "stage": stage, "split": a.split, "pid": os.getpid(),
                "command": args, "updated_unix_seconds": time.time()}
    json_atomic(a.run_dir / "workflow_progress.json", progress)
    try:
        subprocess.run(args, cwd=str(ROOT), env=env, check=True)
    except Exception as exc:
        json_atomic(a.run_dir / "workflow_progress.json", dict(progress, status="failed", error=str(exc),
                    elapsed_seconds=time.monotonic() - begun, updated_unix_seconds=time.time()))
        raise
    elapsed = time.monotonic() - begun
    with (a.run_dir / "stage_timings.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, "split": a.split, "wall_seconds": elapsed,
                                 "completed_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    json_atomic(a.run_dir / "workflow_progress.json", dict(progress, status="stage_completed",
                elapsed_seconds=elapsed, updated_unix_seconds=time.time()))


def prepared(a):
    path = a.data_dir / "input_manifest.json"
    if not path.exists():
        command(a, "export_answer_free_inputs", [a.python, ROOT / "scripts/prepare_evidence_search.py", "export",
                "--source-dir", a.source_dir, "--output-dir", a.data_dir])
    manifest = read(path)
    if manifest["source_dir"] != str(a.source_dir.resolve()):
        raise ValueError("export belongs to another source QA directory")
    for split in ("dev", "eval"):
        if file_sha256(a.data_dir / "inputs" / (split + ".jsonl")) != manifest["splits"][split]["input_sha256"]:
            raise ValueError("answer-free search input changed")
        if file_sha256(a.data_dir / "original" / (split + ".jsonl")) != manifest["splits"][split]["qa_sha256"]:
            raise ValueError("original evaluation QA changed")
    return manifest


def search(a):
    if not (a.model / "config.json").is_file():
        raise FileNotFoundError("external model is not downloaded at %s; download it or set EVIDENCE_MODEL first" % a.model)
    prepared(a)
    args = [a.external_python, "-u", ROOT / "scripts/search_video_evidence.py",
            "--input", a.data_dir / "inputs" / (a.split + ".jsonl"), "--model", a.model,
            "--output-dir", search_dir(a), "--split", a.split, "--workers", a.workers,
            "--min-pixels", a.min_pixels, "--max-pixels", a.max_pixels,
            "--total-pixels", a.total_pixels, "--max-input-tokens", a.max_input_tokens,
            "--max-new-tokens", a.search_max_new_tokens]
    if a.split == "eval":
        args.extend(["--frozen-config", a.run_dir / "frozen_search_config.json"])
    if a.retry_failed:
        args.append("--retry-failed")
    command(a, "external_evidence_search", args)


def ensure_views(a):
    prepared(a)
    manifest = read(split_dir(a) / "manifest.json")
    if manifest is None:
        manifest = assemble_views(a.data_dir, search_dir(a), a.split)
    run = read(search_dir(a) / "run_manifest.json")
    if run is None or manifest["search_manifest_sha256"] != file_sha256(search_dir(a) / "run_manifest.json"):
        raise ValueError("view plans belong to another search run")
    expected = {"data.jsonl": "qa_sha256", "views.jsonl": "views_sha256", "regions.jsonl": "catalog_sha256"}
    for filename, key in expected.items():
        if file_sha256(split_dir(a) / filename) != manifest[key]:
            raise ValueError("assembled views/data/crop catalog changed")
    if a.split == "eval":
        frozen = read(a.run_dir / "frozen_search_config.json")
        if frozen is None or frozen["policy_identity"] != run["policy_identity"]:
            raise ValueError("official preprocessing does not use the dev-frozen configuration")
    return manifest


def cache(a):
    ensure_views(a)
    command(a, "cache_selected_eva_crops", [a.python, "-u", "-m", "torch.distributed.run",
        "--nproc_per_node=" + str(a.cache_workers), "--master_port=" + str(a.master_port),
        ROOT / "scripts/cache_region_features.py", "--catalog", split_dir(a) / "regions.jsonl",
        "--cache-dir", cache_dir(a), "--upstream-root", a.upstream, "--checkpoint-root", a.checkpoints,
        "--visual-checkpoint-root", a.visual_checkpoints, "--bridge-init", a.bridge,
        "--batch-size", a.cache_batch_size, "--existing-policy", "validate"])


def evaluate(a):
    manifest = ensure_views(a)
    cache_summary = read(cache_dir(a) / "cache_summary.json")
    if not cache_summary or not cache_summary.get("completed") or cache_summary["catalog_sha256"] != manifest["catalog_sha256"]:
        raise ValueError("selected crop EVA cache is not complete for this catalog; run cache first")
    conditions = ["full", "evidence"] + (["duplicate"] if a.include_duplicate else [])
    weight_identity = mpp_weight_identity(a)
    check_official_mpp_identity(a, weight_identity)
    for condition in conditions:
        folder = eval_dir(a, condition)
        kind = "roi" if condition == "evidence" else condition
        generation = read(folder / "generation_summary.json") or {}
        generation_run = read(folder / "run_manifest.json") or {}
        expected = expected_generation_identity(manifest, kind, weight_identity)
        if generation.get("completed") == manifest["records"] and generation.get("total") == manifest["records"]:
            if any(generation_run.get(k) != v for k, v in expected.items()):
                raise ValueError("completed MPP generation has a different frozen run identity")
            print("Reusing complete MPP answers: " + condition, flush=True)
        else:
            command(a, "mpp_generate_" + condition, [a.python, "-u", ROOT / "scripts/generate_answers.py",
                "--upstream-root", a.upstream, "--checkpoint-root", a.checkpoints,
                "--visual-checkpoint-root", a.visual_checkpoints, "--bridge-init", a.bridge,
                "--feature-stage", "eva", "--precision-protocol", "fp32_trainables_v1",
                "--task-instruction-version", "legacy_v1", "--max-txt-len", "1536", "--qformer-mode", "auto",
                "--adapter", a.mpp_init / "adapter", "--projector", a.mpp_init / "projector.pt",
                "--qformer-adapter", a.mpp_init / "qformer_adapter.pt", "--data", split_dir(a) / "data.jsonl",
                "--cache-dir", a.full_cache, "--region-views", split_dir(a) / "views.jsonl",
                "--region-cache-dir", cache_dir(a), "--region-condition", kind,
                    "--output-dir", folder, "--batch-size", "1", "--max-new-tokens", "96"], single_gpu=True)
        input_identity = {"predictions_sha256": file_sha256(folder / "predictions.jsonl"),
                          "judge_protocol": JUDGE_PROTOCOL}
        if not (read(folder / "scored/input_identity.json") == input_identity and
                read(folder / "scored/summary.json")):
            command(a, "judge_score_" + condition, [a.python, "-u", ROOT / "scripts/evaluate_answers.py",
                    "--predictions", folder / "predictions.jsonl", "--output-dir", folder / "scored",
                    "--lingo-judge"], single_gpu=True)
            json_atomic(folder / "scored/input_identity.json", input_identity)
    return make_report(a)


def mpp_weight_identity(a):
    return {"adapter_identity": fingerprint(a.mpp_init / "adapter"),
            "projector_identity": file_sha256(a.mpp_init / "projector.pt"),
            "qformer_adapter_sha256": file_sha256(a.mpp_init / "qformer_adapter.pt"),
            "bridge_init_sha256": file_sha256(a.bridge), "dtype": "bf16", "qformer_mode": "auto"}


def check_official_mpp_identity(a, weights):
    if a.split == "eval":
        frozen = read(a.run_dir / "frozen_search_config.json")
        if not frozen or frozen.get("mpp_weight_identity") != weights:
            raise ValueError("official evaluation must keep the MPP weights/precision frozen on dev")


def expected_generation_identity(manifest, kind, weights):
    return {"data_sha256": manifest["qa_sha256"], "region_views_sha256": manifest["views_sha256"],
            "region_view_protocol": EVIDENCE_VIEW_PROTOCOL, "evidence_policy_sha256": manifest["policy_sha256"],
            "region_condition": kind, "feature_stage": "eva", "task_instruction_version": "legacy_v1", "max_records": 0,
            "max_new_tokens": 96, "batch_size": 1, "precision_protocol": "fp32_trainables_v1", **weights}


def checked_scores(a, condition, originals, manifest, weight_identity):
    folder = eval_dir(a, condition)
    summary = read(folder / "scored/summary.json")
    if summary is None:
        return None
    run = read(folder / "run_manifest.json") or {}
    expected = expected_generation_identity(manifest, "roi" if condition == "evidence" else condition, weight_identity)
    if any(run.get(k) != v for k, v in expected.items()) or summary.get("judge_protocol") != JUDGE_PROTOCOL:
        raise ValueError("evaluation changed QA, observation plans, MPP weights, or scoring protocol")
    generation = read(folder / "generation_summary.json") or {}
    if generation.get("completed") != len(originals) or generation.get("total") != len(originals):
        raise ValueError("generation is incomplete")
    generated = load_jsonl(folder / "predictions.jsonl")
    scored = load_jsonl(folder / "scored/scored_predictions.jsonl")
    by_id, predictions = {r["question_id"]: r for r in scored}, {r["question_id"]: r for r in generated}
    if (len(by_id) != len(scored) or len(predictions) != len(generated) or
            set(by_id) != set(originals) or set(predictions) != set(originals)):
        raise ValueError("scoring must cover each original question exactly once")
    plans = {p["question_id"]: p for p in load_jsonl(split_dir(a) / "views.jsonl")}
    for qid, row in by_id.items():
        source, prediction, plan = originals[qid], predictions[qid], plans[qid]
        for detail in (row, prediction):
            if (detail["question"] != source["question"] or detail["references"] != source["answers"] or
                    detail["segment_id"] != source["segment_id"]):
                raise ValueError("scoring uses modified questions/references")
        if row["prediction"] != prediction["prediction"] or not isinstance(row["lingo_judge_correct"], bool):
            raise ValueError("scoring is stale or lacks official Judge decisions")
        observation = prediction.get("region_observation", {})
        views = 5 if condition == "full" else len(plan["views"])
        if (observation.get("plan_sha256") != plan["plan_sha256"] or observation.get("views") != views or
                observation.get("condition") != expected["region_condition"] or
                observation.get("evidence_search") != plan["evidence_search"]):
            raise ValueError("generated answer used different evidence images/plan")
    actual = summarize(scored)
    if any(abs(summary.get(k, -1) - v) > 1e-9 for k, v in actual.items()):
        raise ValueError("reported scoring summary differs from its scored rows")
    return scored


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def make_report(a):
    manifest = ensure_views(a)
    original_rows = load_jsonl(split_dir(a) / "data.jsonl")
    originals = {r["question_id"]: r for r in original_rows}
    run = read(search_dir(a) / "run_manifest.json")
    inputs = load_jsonl(a.data_dir / "inputs" / (a.split + ".jsonl"))
    results = []
    for row in inputs:
        result = read(result_file(search_dir(a), row["question_id"]))
        validate_result(result, row, run["policy_sha256"], run["policy_identity"]["config"])
        results.append(result)
    plans = load_jsonl(split_dir(a) / "views.jsonl")
    if any(p["evidence_search"]["result_sha256"] != r["result_sha256"] for p, r in zip(plans, results)):
        raise ValueError("search results changed after freezing the MPP view plans")
    weights = mpp_weight_identity(a)
    check_official_mpp_identity(a, weights)
    names = ["full", "evidence"] + (["duplicate"] if a.include_duplicate else [])
    scored = {n: checked_scores(a, n, originals, manifest, weights) for n in names}
    missing = [n for n, rows in scored.items() if rows is None]
    models, comparisons = {}, {}
    for name, rows in scored.items():
        if rows is None:
            continue
        models[name] = {"combined": summarize(rows), "by_source": {}}
        for source in ("action", "scenery"):
            subset = [r for r in rows if r.get("source_subset") == source]
            if subset:
                models[name]["by_source"][source] = summarize(subset)
    if not missing:
        comparisons["full_vs_evidence"] = paired(scored["full"], scored["evidence"])
        if "duplicate" in scored:
            comparisons["duplicate_vs_evidence"] = paired(scored["duplicate"], scored["evidence"])
        for state in ("selected", "full_only", "failed"):
            ids = {r["question_id"] for r in results if r["status"] == state}
            if ids:
                comparisons["state_" + state] = paired([r for r in scored["full"] if r["question_id"] in ids],
                                                      [r for r in scored["evidence"] if r["question_id"] in ids])
    search_times = [r["elapsed_seconds"] for r in results]
    cache_summary = read(cache_dir(a) / "cache_summary.json")
    report = {"protocol": SEARCH_PROTOCOL, "split": a.split, "completed": not missing,
              "records": len(originals), "missing_mpp_runs": missing, "policy_sha256": run["policy_sha256"],
              "search_model": run["policy_identity"]["model"], "search_config": run["policy_identity"]["config"],
              "search_states": dict(Counter(r["status"] for r in results)),
              "observed_crop_count": sum(len(r["observed_regions"]) for r in results),
              "selected_crop_count": manifest["region_slots"], "unique_encoded_crops": manifest["unique_regions"],
              "models": models, "paired": comparisons, "judge_protocol": JUDGE_PROTOCOL,
              "original_qa_unchanged": True, "external_answers_evaluated": False,
              "external_text_passed_to_mpp": False,
              "costs": {"search_seconds_sum": sum(search_times),
                        "search_seconds_mean_per_question": sum(search_times) / len(search_times),
                        "search_seconds_p50": percentile(search_times, .5), "search_seconds_p95": percentile(search_times, .95),
                        "search_model_calls": sum(len(r["trace"]) for r in results),
                        "crop_feature_cache": cache_summary,
                        "mpp_model_latency_ms_mean": {n: sum(r["latency_ms"] for r in rows) / len(rows)
                                                      for n, rows in scored.items() if rows},
                        "stage_attempt_wall_timings": load_jsonl(a.run_dir / "stage_timings.jsonl")
                            if (a.run_dir / "stage_timings.jsonl").exists() else [],
                        "end_to_end_latency_measured": False,
                        "note": "MPP model latency excludes search and feature preparation. Search times and "
                                "deduplicated batch cache wall time are separate, not a measured online latency."},
              "limitations": ["selected regions are model hypotheses, not human-verified object identities",
                              "frozen MPP has not been trained on these views; benefit is not guaranteed",
                              "official 500 has been reused; it is not a new blind test",
                              "this policy is not ZoomEye reproduction or OPD"]}
    destination = a.run_dir / ("report_" + a.split)
    json_atomic(destination.with_suffix(".json"), report)
    lines = ["# Question-related evidence / frozen MPP — " + a.split, "", "Only MPP full vs MPP evidence is evaluated; "
             "no external direct-QA benchmark or training.", "", "| Input | Judge | Token F1 |", "|---|---:|---:|"]
    for name, value in models.items():
        metric = value["combined"]
        lines.append("| %s | %.2f%% | %.4f |" % (name, 100 * metric["lingo_judge_accuracy"], metric["token_f1"]))
    lines += ["", "Search states: " + json.dumps(report["search_states"]), "",
              "Observed / selected crops: %d / %d; deduplicated encoded crops: %d." %
              (report["observed_crop_count"], report["selected_crop_count"], report["unique_encoded_crops"]), ""]
    for name, comparison in comparisons.items():
        lines.append("%s: fixed %d, regressed %d, net %d; delta %.2f pp; segment-bootstrap 95%% CI %s." %
                     (name, comparison["fixed"], comparison["regressed"], comparison["net"],
                      comparison["net_change_mean"] * 100,
                      [round(x * 100, 3) for x in comparison["segment_bootstrap_95pct"]]))
    lines += ["", "Search mean %.2f s/query; MPP latency excludes search/crop preparation. See JSON costs and "
              "stage timings. No online end-to-end latency claim." % report["costs"]["search_seconds_mean_per_question"],
              "", "Original QA unchanged. Official 500 is reused, not blind. No new model-training result."]
    destination.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def freeze(a):
    if a.split != "dev":
        raise ValueError("freeze is defined using dev only")
    report = make_report(a)
    if not report["completed"]:
        raise ValueError("finish both frozen MPP dev arms before freezing official search")
    run = read(search_dir(a) / "run_manifest.json")
    inputs = prepared(a)
    payload = {"protocol": SEARCH_PROTOCOL, "selection_split": "dev",
               "policy_identity": run["policy_identity"], "policy_sha256": run["policy_sha256"],
               "mpp_weight_identity": mpp_weight_identity(a),
               "dev_report_sha256": file_sha256(a.run_dir / "report_dev.json"),
               "dev_search_states": report["search_states"],
               "dev_comparison": report["paired"]["full_vs_evidence"],
               "official_input_sha256": inputs["splits"]["eval"]["input_sha256"],
               "freeze_is_explicit_user_action": True,
               "official_500_is_new_blind_test": False}
    immutable_json(a.run_dir / "frozen_search_config.json", payload)
    return payload


def status(a):
    output = {"run_dir": str(a.run_dir), "workflow": read(a.run_dir / "workflow_progress.json"),
              "external_model_available": (a.model / "config.json").is_file(), "external_model_path": str(a.model),
              "input_export": read(a.data_dir / "export_progress.json"), "splits": {}}
    for split in ("dev", "eval"):
        folder = a.run_dir / ("search_" + split)
        workers = [read(p) for p in sorted(folder.glob("progress_worker_*.json"))]
        launcher = read(folder / "launcher_progress.json")
        # A new worker count can resume a run; ignore progress from the old count.
        count = launcher.get("workers") if launcher else None
        if count:
            workers = [v for v in workers if v.get("workers") == count]
        for value in workers:
            value["seconds_since_update"] = round(time.time() - value["updated_unix_seconds"], 1)
            try:
                os.kill(value["pid"], 0)
                value["process_alive"] = True
            except (OSError, KeyError):
                value["process_alive"] = False
        compact = [{k: v for k, v in value.items() if k in (
            "worker", "status", "completed", "assigned_total", "question_id", "call", "observed_crops",
            "search_states", "recent_seconds_per_question", "estimated_remaining_seconds", "process_alive",
            "seconds_since_update", "last_error", "traceback")} for value in workers]
        info = {"search_launcher": launcher, "search_summary": read(folder / "search_summary.json"),
                "workers": compact, "search_completed": sum(x["completed"] for x in workers),
                "search_total": sum(x["assigned_total"] for x in workers),
                "assembled_views": read(a.data_dir / split / "manifest.json"),
                "crop_cache_summary": read(a.cache_root / split / "cache_summary.json"),
                "crop_cache_workers": [read(p) for p in sorted((a.cache_root / split).glob("progress_rank_*.json"))],
                "mpp": {}, "report_ready": bool((read(a.run_dir / ("report_%s.json" % split)) or {}).get("completed"))}
        for condition in ("full", "evidence", "duplicate"):
            evaluation = a.run_dir / ("mpp_%s_%s" % (condition, split))
            path = evaluation / "predictions.jsonl"
            generated = 0
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    generated = sum(bool(line.strip()) for line in handle)
            summary = read(evaluation / "scored/summary.json")
            info["mpp"][condition] = {"generated": generated,
                "stage": "scored" if summary else "scoring" if (evaluation / "generation_summary.json").exists()
                         else "generating_or_loading" if (evaluation / "run_manifest.json").exists() else "not_started",
                "score_progress": read(evaluation / "scored/score_progress.json")}
        output["splits"][split] = info
    return output


def render_status(value):
    """A watch-friendly screen, not hundreds of lines of nested JSON."""
    lines = ["Run: " + value["run_dir"]]
    workflow = value["workflow"] or {}
    stage = workflow.get("stage", workflow.get("action", "not_started"))
    lines.append("Workflow: %s / %s (%s)" % (workflow.get("split", "-"), stage, workflow.get("status", "not_started")))
    if workflow.get("error"):
        lines.append("ERROR: " + workflow["error"])
    lines.append("External model: %s [%s]" % (value["external_model_path"],
                 "local config present" if value["external_model_available"] else "not downloaded here"))
    export = value["input_export"] or {}
    lines.append("Input export: %s %s/%s" % (export.get("status", "not_started"),
                 export.get("completed", "-"), export.get("total", "-")))
    for split, info in value["splits"].items():
        launcher = info["search_launcher"] or {}
        lines.append("%s search: %d/%d [%s]" % (split.upper(), info["search_completed"], info["search_total"],
                     launcher.get("status", "not_started")))
        for worker in info["workers"]:
            eta = worker.get("estimated_remaining_seconds")
            lines.append("  GPU worker %d: %s %d/%d call=%s crops=%s ETA=%s alive=%s age=%.1fs" %
                         (worker["worker"], worker["status"], worker["completed"], worker["assigned_total"],
                          worker.get("call", "-"), worker.get("observed_crops", "-"),
                          "%.1fmin" % (eta / 60) if eta is not None else "-",
                          worker.get("process_alive"), worker["seconds_since_update"]))
            if worker.get("last_error"):
                lines.append("    failed question: " + worker["last_error"]["message"])
            if worker.get("traceback"):
                lines.append("    ERROR: " + worker["traceback"].strip().splitlines()[-1])
        cache = info["crop_cache_summary"] or {}
        ranks = info["crop_cache_workers"]
        lines.append("  EVA crop cache: %s %d/%s" % ("completed" if cache.get("completed") else
                     ",".join(r["status"] for r in ranks) or "not_started",
                     sum(r.get("completed", 0) for r in ranks),
                     cache.get("regions", sum(r.get("assigned_total", 0) for r in ranks))))
        for name, mpp in info["mpp"].items():
            if name == "duplicate" and mpp["stage"] == "not_started":
                continue
            scoring = mpp["score_progress"] or {}
            lines.append("  MPP %s: generated=%d stage=%s scored=%s/%s" %
                         (name, mpp["generated"], mpp["stage"], scoring.get("completed", "-"), scoring.get("total", "-")))
        lines.append("  report ready: " + str(info["report_ready"]))
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "search", "assemble", "cache", "evaluate", "run", "freeze", "report", "status", "preview"))
    p.add_argument("--split", choices=("dev", "eval"), default="dev")
    p.add_argument("--source-dir", type=Path, default=ROOT / "datasets/processed/lingoqa_feedback_opd_40k_v2")
    p.add_argument("--data-dir", type=Path, default=ROOT / "datasets/processed/lingoqa_vlm_evidence_v1")
    p.add_argument("--run-dir", type=Path, default=ROOT / "outputs/vlm_evidence_search_v1")
    p.add_argument("--cache-root", type=Path, default=ROOT / "datasets/cache/lingoqa_vlm_evidence_eva_v1")
    p.add_argument("--model", type=Path, default=Path("/data/zjr/models/Qwen3-VL-8B-Instruct"))
    p.add_argument("--python", default="/home/zjr/anaconda3/envs/mpp_llava/bin/python")
    p.add_argument("--external-python", default="/home/zjr/anaconda3/envs/chartvlm/bin/python")
    p.add_argument("--upstream", type=Path, default=ROOT.parent / "mpp_llava_video_workspace/third_party/MPP-LLaVA")
    p.add_argument("--checkpoints", type=Path, default=Path("/tmp/zjr_mpp_fast/checkpoints"))
    p.add_argument("--visual-checkpoints", type=Path, default=ROOT.parent / "mpp_llava_video_workspace/checkpoints")
    p.add_argument("--bridge", type=Path, default=ROOT / "checkpoints/visual_bridge_init.pt")
    p.add_argument("--full-cache", type=Path, default=Path("/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1"))
    p.add_argument("--mpp-init", type=Path, default=ROOT / "outputs/mpp_feedback_sft_control_40k_2epoch_v2/final")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-workers", type=int, default=4)
    p.add_argument("--cache-batch-size", type=int, default=4)
    p.add_argument("--master-port", type=int, default=29712)
    p.add_argument("--min-pixels", type=int, default=50176)
    p.add_argument("--max-pixels", type=int, default=200704)
    p.add_argument("--total-pixels", type=int, default=2007040)
    p.add_argument("--max-input-tokens", type=int, default=6144)
    p.add_argument("--search-max-new-tokens", type=int, default=512)
    p.add_argument("--include-duplicate", action="store_true", help="optional extra MPP input control, not external QA")
    p.add_argument("--retry-failed", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    # Resolve all paths so launching from Tracker rather than the project works.
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve())
    if not (a.checkpoints / "mpp-qwen-next-sft/sharded/model.safetensors.index.json").exists():
        a.checkpoints = a.visual_checkpoints
    if a.action != "status":
        json_atomic(a.run_dir / "workflow_progress.json", {"status": "initializing", "action": a.action,
                    "split": a.split, "pid": os.getpid(), "updated_unix_seconds": time.time()})
    if a.action == "status":
        value = status(a)
        print(render_status(value), flush=True)
        return
    elif a.action == "prepare":
        value = export_inputs(a.source_dir, a.data_dir)
    elif a.action == "search":
        search(a); value = read(search_dir(a) / "search_summary.json")
    elif a.action == "assemble":
        value = assemble_views(a.data_dir, search_dir(a), a.split)
    elif a.action == "cache":
        cache(a); value = read(cache_dir(a) / "cache_summary.json")
    elif a.action == "evaluate":
        value = evaluate(a)
    elif a.action == "report":
        value = make_report(a)
    elif a.action == "freeze":
        value = freeze(a)
    elif a.action == "preview":
        ensure_views(a)
        command(a, "html_preview", [a.python, ROOT / "scripts/preview_region_views.py", "--data",
                split_dir(a) / "data.jsonl", "--views", split_dir(a) / "views.jsonl", "--output",
                a.run_dir / ("evidence_preview_%s.html" % a.split), "--count", "20"])
        value = {"html": str(a.run_dir / ("evidence_preview_%s.html" % a.split))}
    else:
        if a.split == "eval" and not (a.run_dir / "frozen_search_config.json").is_file():
            raise ValueError("run dev and explicitly freeze its policy before official evaluation")
        search(a)
        assemble_views(a.data_dir, search_dir(a), a.split)
        cache(a)
        value = evaluate(a)
    if a.action != "status":
        json_atomic(a.run_dir / "workflow_progress.json", {"status": "completed", "action": a.action,
                    "split": a.split, "updated_unix_seconds": time.time()})
    print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        a = parse_args()
        if a.action != "status":
            json_atomic(a.run_dir / "workflow_progress.json", {"status": "failed", "action": a.action,
                        "split": a.split, "error": str(exc), "updated_unix_seconds": time.time()})
        raise
