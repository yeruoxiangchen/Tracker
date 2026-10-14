#!/usr/bin/env python3
"""Cache a fixed joint model's no-hint response logps; no training or sampling."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import file_sha256, load_jsonl
from mpp_video_qa.preference import (PreferenceDataset, REFERENCE_PROTOCOL, cache_inventory,
                                    encode_completion, completion_logprob, tokenizer_identity)
from scripts.generate_answers import write_json_atomic
from scripts.preference_common import add_model_arguments, make_model, validate_selection


def parse_args():
    p = argparse.ArgumentParser()
    add_model_arguments(p)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = PreferenceDataset(args.data, args.cache_dir)
    _, init = validate_selection(args, dataset.rows)
    progress = lambda **kw: write_json_atomic(args.output_dir / "progress.json", dict(updated_unix_seconds=time.time(), **kw))
    progress(status="checking_selected_cache", total=len(dataset))
    inventory = cache_inventory(dataset.rows, args.cache_dir)
    identity = {"protocol": REFERENCE_PROTOCOL, "data_sha256": file_sha256(args.data),
                "initialization": init, "visual_cache_sha256": inventory,
                "answer_tokens": "canonical text tokenization plus exactly one im_end; sum logp"}
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(old.get(k) != v for k, v in identity.items()):
            raise ValueError("reference directory belongs to a different model/data/cache")
    destination = args.output_dir / "reference.jsonl"
    existing = load_jsonl(destination) if destination.exists() else []
    done = {r["example_id"]: r for r in existing}
    expected = {r["example_id"]: r for r in dataset.rows}
    if len(done) != len(existing) or not set(done).issubset(expected):
        raise ValueError("invalid existing reference records")
    if existing and not manifest_path.exists():
        raise ValueError("reference records have no identity manifest")
    for key, row in done.items():
        if row["row_sha256"] != expected[key]["row_sha256"]:
            raise ValueError("existing reference row differs")
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if len(done) != len(dataset) or summary["reference_sha256"] != file_sha256(destination):
            raise ValueError("completed reference cache was modified")
        progress(status="completed", completed=len(done), total=len(dataset))
        print(json.dumps(summary, indent=2))
        return
    progress(status="loading_reference", completed=len(done), total=len(dataset))
    # Match policy arithmetic (bf16 frozen base, fp32 adapters/projector), then
    # freeze the reference. A bf16-only projector would change initial logps.
    model = make_model(args, True, torch.device("cuda")).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = model.backbone.llm_tokenizer
    identity["tokenizer_sha256"] = tokenizer_identity(tokenizer)
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
        raise ValueError("reference tokenizer changed")
    write_json_atomic(manifest_path, identity)
    started = time.time()
    with destination.open("a", encoding="utf-8") as stream, torch.no_grad():
        for index, row in enumerate(dataset.rows):
            if row["example_id"] in done:
                continue
            features = dataset[index]["features"].unsqueeze(0).cuda()
            record = {"example_id": row["example_id"], "row_sha256": row["row_sha256"]}
            for kind in ("chosen", "rejected") if row["kind"] == "preference" else ("chosen",):
                ids = encode_completion(tokenizer, row[kind])
                value = completion_logprob(model, features, row["question"], ids, args.chunk_tokens)
                if not torch.isfinite(value):
                    raise ValueError("nonfinite reference score")
                record[kind + "_ids"] = ids
                record["ref_" + kind + "_logp"] = float(value)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            done[row["example_id"]] = record
            progress(status="scoring_reference", completed=len(done), total=len(dataset), elapsed_seconds=time.time() - started)
            print(json.dumps({"reference_completed": len(done), "total": len(dataset)}), flush=True)
    summary = {"completed": True, "records": len(done), "reference_sha256": file_sha256(destination),
               "elapsed_seconds_this_run": time.time() - started,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    write_json_atomic(summary_path, summary)
    progress(status="completed", completed=len(done), total=len(dataset))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        args = parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.output_dir / "progress.json", {
            "status": "failed", "traceback": traceback.format_exc(), "updated_unix_seconds": time.time()})
        raise
