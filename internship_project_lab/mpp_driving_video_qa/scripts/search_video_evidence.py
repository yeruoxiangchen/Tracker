#!/usr/bin/env python3
"""Independent per-GPU question shards: external VLM -> actual crops -> evidence.

No torch.distributed, NCCL or references. Each question has an atomic result and
trace; the same frozen policy/input can resume using a different worker count.
"""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl
from mpp_video_qa.evidence_search import (DEFAULT_CONFIG, SEARCH_PROTOCOL, SYSTEM_PROMPT, content_hash,
    result_file, search_one, validate_config, validate_input, validate_result)
from mpp_video_qa.region_views import json_atomic
from scripts.generate_vlm_teacher import model_identity
from scripts.prepare_evidence_search import immutable_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="EXPORTED answer-free JSONL, not original QA")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", choices=("dev", "eval"), required=True)
    p.add_argument("--frozen-config", type=Path, help="required before official eval search")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--worker-index", type=int, default=-1, help=argparse.SUPPRESS)
    p.add_argument("--max-records", type=int, default=0, help="separate-directory interface check only")
    p.add_argument("--retry-failed", action="store_true", help="explicitly recompute failed questions only")
    for key in ("min_pixels", "max_pixels", "total_pixels", "max_input_tokens", "max_new_tokens"):
        p.add_argument("--" + key.replace("_", "-"), type=int, default=DEFAULT_CONFIG[key])
    return p.parse_args()


def policy_identity(args):
    config = dict(DEFAULT_CONFIG)
    for key in ("min_pixels", "max_pixels", "total_pixels", "max_input_tokens", "max_new_tokens"):
        config[key] = getattr(args, key)
    validate_config(config)
    return {"protocol": SEARCH_PROTOCOL, "model": model_identity(args.model), "config": config,
            "system_prompt": SYSTEM_PROMPT, "decoding": "greedy", "dtype": "bf16",
            "attention": "sdpa", "input_mode": "chronological_images_and_returned_native_crops",
            "external_text_passed_to_mpp": False, "answers_or_scores_in_search": False,
            "software": {n: importlib.metadata.version(n) for n in ("transformers", "torch", "Pillow")},
            "implementation_sha256": {n: file_sha256(ROOT / n) for n in (
                "mpp_video_qa/evidence_search.py", "mpp_video_qa/evidence_hf.py",
                "mpp_video_qa/core.py", "mpp_video_qa/region_views.py",
                "scripts/generate_vlm_teacher.py", "scripts/search_video_evidence.py")}}


def initialize(args):
    if args.workers < 1 or args.max_records < 0:
        raise ValueError("workers must be positive and max-records nonnegative")
    rows = load_jsonl(args.input)
    for row in rows:
        validate_input(row)
    if not rows or len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("answer-free search inputs must be nonempty and have unique question IDs")
    policy = policy_identity(args)
    if args.split == "eval":
        if args.frozen_config is None:
            raise ValueError("official search requires the configuration frozen after dev evaluation")
        frozen = json.loads(args.frozen_config.read_text())
        if (frozen["policy_identity"] != policy or frozen["official_input_sha256"] != file_sha256(args.input) or
                args.max_records != 0):
            raise ValueError("official search differs from the dev-frozen policy or official input")
    identity = {"protocol": SEARCH_PROTOCOL, "split": args.split,
                "input_sha256": file_sha256(args.input), "input_path": str(args.input.resolve()),
                "input_records": len(rows), "max_records": args.max_records,
                "policy_identity": policy, "policy_sha256": content_hash(policy)}
    immutable_json(args.output_dir / "run_manifest.json", identity)
    if args.max_records:
        rows = rows[:args.max_records]
    return rows, identity


def worker(args, rows, identity):
    rank = args.worker_index if args.worker_index >= 0 else 0
    assigned = rows[rank::args.workers]
    lock_path = args.output_dir / ("worker_%d.lock" % rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        # Do not accidentally run the same question shard twice in two launchers.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started, durations, done, states = time.monotonic(), [], 0, {"selected": 0, "full_only": 0, "failed": 0}
        def progress(stage, **extra):
            recent = sum(durations[-20:]) / len(durations[-20:]) if durations else None
            payload = {"status": stage, "worker": rank, "workers": args.workers, "pid": os.getpid(),
                       "completed": done, "assigned_total": len(assigned), "records_total": len(rows),
                       "search_states": dict(states), "elapsed_seconds": time.monotonic() - started,
                       "recent_seconds_per_question": recent,
                       "estimated_remaining_seconds": (len(assigned) - done) * recent if recent else None,
                       "updated_unix_seconds": time.time(), **extra}
            json_atomic(args.output_dir / ("progress_worker_%d.json" % rank), payload)
            print(json.dumps(payload), flush=True)
        try:
            progress("validating_resume")
            pending = []
            for row in assigned:
                path = result_file(args.output_dir, row["question_id"])
                if path.exists():
                    old = json.loads(path.read_text())
                    validate_result(old, row, identity["policy_sha256"], identity["policy_identity"]["config"])
                    if old["status"] != "failed" or not args.retry_failed:
                        done += 1
                        states[old["status"]] += 1
                        continue
                pending.append(row)
            if not pending:
                progress("completed")
                return
            progress("loading_external_vlm")
            from mpp_video_qa.evidence_hf import HFSearchBackend
            backend = HFSearchBackend(args.model, identity["policy_identity"]["config"])
            progress("model_loaded")
            for row in pending:
                def event(stage, **extra):
                    progress(stage, question_id=row["question_id"], **extra)
                result = search_one(row, backend, identity["policy_identity"]["config"],
                                    identity["policy_sha256"], event)
                validate_result(result, row, identity["policy_sha256"], identity["policy_identity"]["config"])
                json_atomic(result_file(args.output_dir, row["question_id"]), result)
                done += 1
                states[result["status"]] += 1
                durations.append(result["elapsed_seconds"])
                progress("searching", question_id=row["question_id"], last_search_status=result["status"],
                         last_error=result["error"])
                error = result["error"] or {}
                if ("OutOfMemory" in error.get("type", "") or "CUDA" in error.get("message", "") or
                        "CUBLAS" in error.get("message", "")):
                    raise RuntimeError("external VLM GPU failure; result saved. Adjust capacity in a new run, "
                                       "or resume unchanged with --retry-failed after freeing the GPU")
            progress("completed")
        except Exception:
            progress("worker_failed", traceback=traceback.format_exc())
            raise


def run_search(args):
    # Emit startup state before any heavyweight Transformers/Torch import.
    if args.worker_index < 0:
        json_atomic(args.output_dir / "launcher_progress.json", {"status": "initializing", "pid": os.getpid(),
                    "workers": args.workers, "updated_unix_seconds": time.time()})
    rows, identity = initialize(args)
    if args.worker_index >= 0 or args.workers == 1:
        worker(args, rows, identity)
        if args.worker_index >= 0:
            return  # Other independent shards may still be running.
    else:
        gpus = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
        if len(gpus) != args.workers:
            raise ValueError("set CUDA_VISIBLE_DEVICES to exactly one GPU per independent search worker")
        children = []
        try:
            for rank, gpu in enumerate(gpus):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="1")
                command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:],
                           "--worker-index", str(rank)]
                children.append(subprocess.Popen(command, env=env, cwd=str(ROOT)))
            json_atomic(args.output_dir / "launcher_progress.json", {"status": "running", "pid": os.getpid(),
                        "workers": args.workers, "child_pids": [p.pid for p in children],
                        "updated_unix_seconds": time.time()})
            while True:
                codes = [p.poll() for p in children]
                if any(code not in (None, 0) for code in codes):
                    raise RuntimeError("an evidence-search worker failed; see progress_worker_*.json")
                if all(code is not None for code in codes):
                    break
                time.sleep(1)
            if any(p.returncode != 0 for p in children):
                raise RuntimeError("an evidence-search worker failed")
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                child.wait()
    states = {"selected": 0, "full_only": 0, "failed": 0}
    seconds = calls = crops = 0
    for row in rows:
        result = json.loads(result_file(args.output_dir, row["question_id"]).read_text())
        validate_result(result, row, identity["policy_sha256"], identity["policy_identity"]["config"])
        states[result["status"]] += 1
        seconds += result["elapsed_seconds"]
        calls += len(result["trace"])
        crops += len(result["observed_regions"])
    summary = {"completed": True, "records": len(rows), "search_states": states,
               "vlm_calls": calls, "observed_crops": crops, "question_seconds_sum": seconds,
               "policy_sha256": identity["policy_sha256"], "updated_unix_seconds": time.time()}
    if args.worker_index < 0:
        json_atomic(args.output_dir / "search_summary.json", summary)
        json_atomic(args.output_dir / "launcher_progress.json", dict(summary, status="completed", workers=args.workers))
        print(json.dumps(summary, indent=2), flush=True)


def main():
    args = parse_args()
    if args.worker_index >= 0:
        return run_search(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "launcher.lock").open("a") as launcher_lock:
        fcntl.flock(launcher_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_search(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        a = parse_args()
        if a.worker_index < 0 and not isinstance(exc, BlockingIOError):
            json_atomic(a.output_dir / "launcher_progress.json", {"status": "failed", "pid": os.getpid(),
                        "traceback": traceback.format_exc(), "updated_unix_seconds": time.time()})
        raise
