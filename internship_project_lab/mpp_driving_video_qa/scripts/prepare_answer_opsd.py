#!/usr/bin/env python3
"""Prepare original train-only QA for full-frame answer-conditioned OPD probes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, file_sha256
from scripts.prepare_visual_adaptation import stratified_subset, write_rows


def prepare_rows(train, held_out, size, seed=42, per_segment=2):
    if {r["segment_id"] for r in train} & {r["segment_id"] for r in held_out}:
        raise ValueError("train/dev/eval segment overlap")
    if len({r["question_id"] for r in train}) != len(train):
        raise ValueError("duplicate training question IDs")
    selected = stratified_subset(train, size, seed, per_segment)
    if any(len(r["frames"]) != 5 or not r["answers"] or not str(r["answers"][0]).strip()
           for r in selected):
        raise ValueError("expected five original frames and a nonempty first reference")
    return [dict(r, opd_training_source="train") for r in selected]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", default=str(ROOT / "datasets/processed/lingoqa_full_v2"))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-qa-per-segment", type=int, default=2)
    args = p.parse_args()
    source, output = Path(args.source_dir), Path(args.output_dir)
    train = load_jsonl(source / "train.jsonl")
    dev, evaluation = load_jsonl(source / "dev.jsonl"), load_jsonl(source / "eval.jsonl")
    rows = prepare_rows(train, dev + evaluation, args.size, args.seed, args.max_qa_per_segment)
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "probe.jsonl", rows)
    for split in ("dev_action", "dev_scenery", "eval"):
        write_rows(output / (split + ".jsonl"), load_jsonl(source / (split + ".jsonl")))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_train_sha256": file_sha256(source / "train.jsonl"),
        "probe_sha256": file_sha256(output / "probe.jsonl"),
        "held_out_sha256": {name: file_sha256(source / (name + ".jsonl")) for name in ("dev", "eval")},
        "records": len(rows), "segments": len({r["segment_id"] for r in rows}),
        "sources": dict(Counter(r["source_subset"] for r in rows)),
        "seed": args.seed, "max_qa_per_segment": args.max_qa_per_segment,
        "references": "unchanged; teacher receives first reference; student receives none",
        "held_out_segment_overlap": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
