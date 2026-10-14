#!/usr/bin/env python3
"""Immutable original-QA train pool, train-only probe and copied held-out splits."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, file_sha256
from mpp_video_qa.data import cache_file
from scripts.prepare_visual_adaptation import stratified_subset, write_rows
from scripts.generate_answers import write_json_atomic


def prepare(args):
    source, output = Path(args.source_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = load_jsonl(source / "train.jsonl")
    splits = {name: load_jsonl(source / (name + ".jsonl"))
              for name in ("dev", "dev_action", "dev_scenery", "eval")}
    held = splits["dev"] + splits["eval"]
    if ({r["segment_id"] for r in train} & {r["segment_id"] for r in held} or
            {r["question_id"] for r in train} & {r["question_id"] for r in held} or
            {p for r in train for p in r["frames"]} & {p for r in held for p in r["frames"]}):
        raise ValueError("training and held-out video/question/frame overlap")
    if len({r["question_id"] for r in train}) != len(train):
        raise ValueError("duplicate source training question IDs")
    if args.max_qa_per_segment < 0:
        raise ValueError("max-qa-per-segment must be >= 0")
    adaptive = args.max_qa_per_segment == 0
    cap = max(Counter(r["segment_id"] for r in train).values()) if adaptive else args.max_qa_per_segment
    selected = stratified_subset(train, args.train_size, args.seed, cap)
    if any(len(r["frames"]) != 5 or not r["answers"] or not str(r["answers"][0]).strip() for r in selected):
        raise ValueError("five full frames and a nonempty original answer are required")
    selected = [dict(r, feedback_training_source="train") for r in selected]
    probe = stratified_subset(selected, min(args.probe_size, len(selected)), args.seed + 1,
                              cap)
    missing = [r["segment_id"] for r in selected if not cache_file(args.cache_dir, r["segment_id"]).is_file()]
    if missing:
        raise ValueError("missing EVA caches for %d selected records; first=%s" % (len(missing), missing[0]))
    for name, rows in dict(splits, train=selected, probe=probe).items():
        write_rows(output / (name + ".jsonl"), rows)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "records": len(selected),
                "probe_records": len(probe), "seed": args.seed,
                "segments": len({r["segment_id"] for r in selected}),
                "sources": dict(Counter(r["source_subset"] for r in selected)),
                "max_qa_per_segment": args.max_qa_per_segment,
                "source_train_sha256": file_sha256(source / "train.jsonl"),
                "split_sha256": {name: file_sha256(output / (name + ".jsonl"))
                                 for name in ("train", "probe", *splits)},
                "original_qa_preserved": True, "held_out_overlap": 0,
                "cache_dir": str(Path(args.cache_dir).resolve()),
                "probe_scope": "subset of train only; no dev/test labels used"}
    if adaptive:
        counts = Counter(r["segment_id"] for r in selected)
        manifest.update(sampling="source_stratified_video_round_robin_v2",
                        source_segments={s: len({r["segment_id"] for r in selected if r["source_subset"] == s})
                                         for s in sorted(manifest["sources"])},
                        actual_max_qa_per_segment=max(counts.values()),
                        qa_per_segment_distribution={str(k): v for k, v in sorted(Counter(counts.values()).items())},
                        adaptive_max_qa_per_segment=True)
    path = output / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ("created_utc",):
            manifest[key] = previous[key]
        if previous != manifest:
            raise ValueError("existing prepared pool differs; use a new directory")
    write_json_atomic(path, manifest)
    print(json.dumps(manifest, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", default=str(ROOT / "datasets/processed/lingoqa_qformer_all_v1"))
    p.add_argument("--output-dir", default=str(ROOT / "datasets/processed/lingoqa_feedback_opd_v1"))
    p.add_argument("--cache-dir", default="/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1")
    p.add_argument("--train-size", type=int, default=10000)
    p.add_argument("--probe-size", type=int, default=2000)
    p.add_argument("--max-qa-per-segment", type=int, default=2,
                   help="0 uses all rounds needed for source quotas, prioritizing distinct videos")
    p.add_argument("--seed", type=int, default=44)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
