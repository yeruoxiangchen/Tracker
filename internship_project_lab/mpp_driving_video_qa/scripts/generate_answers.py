#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpp_video_qa.data import CachedQADataset, qa_collate
from mpp_video_qa.model import (FEATURE_STAGE_PROJECTED, FEATURE_STAGES,
                                CachedVideoQA, build_text_model)
from mpp_video_qa.core import PROTOCOL, TASK_INSTRUCTIONS, file_sha256
from mpp_video_qa.visual_bridge import add_bridge_arguments
from mpp_video_qa.region_views import add_region_arguments, region_runtime_identity
from mpp_video_qa.question_qformer import question_identity_from_args


def directory_fingerprint(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError("adapter directory does not exist: %s" % path)
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise RuntimeError("adapter directory is empty: %s" % path)
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(file_sha256(item).encode("ascii"))
    return digest.hexdigest()


def write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    add_bridge_arguments(parser)
    add_region_arguments(parser)
    parser.add_argument("--data", required=True); parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--adapter", default="")
    parser.add_argument("--projector", default="")
    parser.add_argument("--feature-stage", choices=FEATURE_STAGES,
                        default=FEATURE_STAGE_PROJECTED)
    parser.add_argument("--upstream-root", required=True); parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-records", type=int, default=0,
                        help="Generate at most this many records; 0 evaluates the full split.")
    parser.add_argument("--max-txt-len", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--lora-r", type=int, default=16); parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["c_attn"])
    parser.add_argument("--task-instruction-version", choices=sorted(TASK_INSTRUCTIONS),
                        default="legacy_v1")
    args = parser.parse_args()
    if args.region_views and (args.feature_stage != "eva" or args.batch_size != 1):
        parser.error("variable automatic views require --feature-stage eva --batch-size 1")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    destination = output / "predictions.jsonl"
    adapter_identity = (directory_fingerprint(args.adapter) if args.adapter else
                        "official_mpp:%s" % Path(args.checkpoint_root).resolve())
    projector_identity = (file_sha256(args.projector) if args.projector else
                          "official_mpp_projection")
    identity = {
        "protocol": PROTOCOL,
        "data_path": str(Path(args.data).resolve()),
        "data_sha256": file_sha256(args.data),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "adapter_path": str(Path(args.adapter).resolve()) if args.adapter else None,
        "adapter_identity": adapter_identity,
        "projector_path": str(Path(args.projector).resolve()) if args.projector else None,
        "projector_identity": projector_identity,
        "feature_stage": args.feature_stage,
        "task_instruction_version": args.task_instruction_version,
        "task_instruction": TASK_INSTRUCTIONS[args.task_instruction_version],
        "max_new_tokens": args.max_new_tokens,
        "max_records": args.max_records,
        "batch_size": args.batch_size,
    }
    if args.feature_stage == "eva":
        identity.update(bridge_init_sha256=file_sha256(args.bridge_init),
                        qformer_adapter_sha256=(file_sha256(args.qformer_adapter)
                                                if args.qformer_adapter else None))
    if args.precision_protocol != "legacy":
        identity.update(precision_protocol=args.precision_protocol, dtype=args.dtype,
                        qformer_mode=args.qformer_mode)
    identity.update(region_runtime_identity(args))
    question_identity = question_identity_from_args(args)
    if question_identity:
        identity["question_qformer"] = question_identity
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != identity:
            raise RuntimeError("output directory belongs to a different generation run: %s" % output)
    elif destination.exists() and destination.stat().st_size:
        raise RuntimeError("predictions exist without run_manifest.json; use a new output directory")
    else:
        write_json_atomic(manifest_path, identity)

    dataset = CachedQADataset(args.data, args.cache_dir, args.feature_stage,
                              region_views=args.region_views, region_cache_dir=args.region_cache_dir,
                              region_condition=args.region_condition,
                              task_instruction=TASK_INSTRUCTIONS[args.task_instruction_version])
    total = min(len(dataset), args.max_records) if args.max_records > 0 else len(dataset)
    target_rows = dataset.rows[:total]
    target_ids = [row["question_id"] for row in target_rows]
    if len(target_ids) != len(set(target_ids)):
        raise RuntimeError("question_id must be unique inside an evaluation run")
    target_id_set = set(target_ids)
    completed = set()
    if destination.is_file():
        for line in destination.read_text(encoding="utf-8").splitlines():
            if line.strip():
                question_id = json.loads(line)["question_id"]
                if question_id not in target_id_set:
                    raise RuntimeError("prediction contains a question outside this run: %s" % question_id)
                if question_id in completed:
                    raise RuntimeError("duplicate prediction for question_id=%s" % question_id)
                completed.add(question_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=qa_collate)
    device = torch.device("cuda")
    model = CachedVideoQA(
        build_text_model(args, train=False, adapter=args.adapter, projector=args.projector),
        task_instruction=TASK_INSTRUCTIONS[args.task_instruction_version],
        feature_stage=args.feature_stage,
    ).to(device).eval()
    bridge = getattr(model.backbone, "visual_bridge", None)
    if getattr(bridge, "question_identity", None) != identity.get("question_qformer"):
        raise ValueError("loaded question bridge differs from generation metadata; include its .question.json sidecar")
    torch.cuda.reset_peak_memory_stats(device)
    done = len(completed); latencies = []
    with open(destination, "a", encoding="utf-8") as handle:
        for batch in loader:
            if done >= total:
                break
            keep = [i for i, row in enumerate(batch["rows"]) if row["question_id"] not in completed]
            keep = keep[:max(0, total - done)]
            if not keep: continue
            features = batch["features"][keep].to(device)
            rows = [batch["rows"][i] for i in keep]
            torch.cuda.synchronize(); start = time.perf_counter()
            prompts = [batch["prompts"][i] for i in keep] if "prompts" in batch else None
            predictions = model.generate(features, [x["question"] for x in rows], args.max_new_tokens,
                                         prompts=prompts)
            torch.cuda.synchronize(); elapsed = 1000 * (time.perf_counter() - start) / len(rows)
            for row, prediction, batch_index in zip(rows, predictions, keep):
                result = {"question_id": row["question_id"], "segment_id": row["segment_id"],
                          "source_subset": row.get("source_subset"),
                          "question": row["question"], "references": row["answers"],
                          "prediction": prediction, "latency_ms": elapsed}
                if "region_metadata" in batch:
                    result["region_observation"] = batch["region_metadata"][batch_index]
                if "diagnostic" in row:
                    result["diagnostic"] = row["diagnostic"]
                handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
                completed.add(row["question_id"]); done += 1; latencies.append(elapsed)
                print(json.dumps({"completed": done, "total": total,
                                  "question_id": row["question_id"]}), flush=True)
    summary = {"completed": len(completed & target_id_set), "total": total,
               "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
               "projector": str(Path(args.projector).resolve()) if args.projector else None,
               "feature_stage": args.feature_stage,
               "precision_protocol": args.precision_protocol,
               "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
               "task_instruction_version": args.task_instruction_version,
               "task_instruction": TASK_INSTRUCTIONS[args.task_instruction_version],
               "latency_ms_mean_new": sum(latencies) / len(latencies) if latencies else None}
    write_json_atomic(output / "generation_summary.json", summary)


if __name__ == "__main__":
    main()
