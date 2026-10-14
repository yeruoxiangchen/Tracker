#!/usr/bin/env python3
"""Coverage-first original-QA subset; no dependency on teacher diagnostics."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, file_sha256
from scripts.prepare_visual_adaptation import stratified_subset, write_rows


def coverage_subset(rows, size, seed=42):
    if not 0 < size <= len(rows):
        raise ValueError("requested size exceeds the original training split")
    by_source = defaultdict(Counter)
    for row in rows:
        by_source[row["source_subset"]][row["segment_id"]] += 1
    # Find a cap that can supply each proportional quota, including rounding.
    cap = 1
    while any(sum(min(n, cap) for n in counts.values()) <
              min(sum(counts.values()), int(size * sum(counts.values()) / len(rows)) + 1)
              for counts in by_source.values()):
        cap += 1
    return stratified_subset(rows, size, seed, cap), cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=ROOT / "datasets/processed/lingoqa_full_v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/processed/lingoqa_qformer_full100k_v1")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--train-size", type=int, default=100000)
    selection.add_argument("--all-train", action="store_true",
                           help="retain every original training row, in source order, without a per-video cap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-cache", type=Path,
                        help="also export 64 long-text train rows whose EVA cache already exists")
    parser.add_argument("--smoke-output", type=Path, default=ROOT / "datasets/processed/lingoqa_qformer_full_smoke_v1")
    args = parser.parse_args()
    source = load_jsonl(args.source_dir / "train.jsonl")
    splits = {name: load_jsonl(args.source_dir / (name + ".jsonl"))
              for name in ("dev", "dev_action", "dev_scenery", "eval")}
    train_segments = {r["segment_id"] for r in source}
    if train_segments & {r["segment_id"] for r in splits["dev"] + splits["eval"]}:
        raise ValueError("source train and held-out segments overlap")
    if args.all_train:
        selected, cap = source, None
    else:
        selected, cap = coverage_subset(source, args.train_size, args.seed)
    if len({r["question_id"] for r in selected}) != len(selected):
        raise ValueError("duplicate selected question IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in dict(splits, train=selected).items():
        write_rows(args.output_dir / (name + ".jsonl"), rows)
    segments = Counter(r["segment_id"] for r in selected)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "records": len(selected), "segments": len(segments), "seed": args.seed,
                "source_counts": dict(Counter(r["source_subset"] for r in selected)),
                "source_segments": {s: len({r["segment_id"] for r in selected if r["source_subset"] == s})
                                    for s in sorted({r["source_subset"] for r in selected})},
                "adaptive_max_qa_per_segment": cap, "actual_max_qa_per_segment": max(segments.values()),
                "sampling": ("all_original_training_rows" if args.all_train else
                             "source_proportional_round_robin_over_distinct_segments"),
                "question_and_answers": "unchanged originals; no generated targets",
                "source_train_sha256": file_sha256(args.source_dir / "train.jsonl"),
                "split_sha256": {n: file_sha256(args.output_dir / (n + ".jsonl"))
                                 for n in ("train", "dev", "dev_action", "dev_scenery", "eval")}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if args.smoke_cache:
        from mpp_video_qa.data import cache_file
        smoke = []
        for row in sorted(selected, key=lambda r: len(r["question"]) + len(r["answers"][0]), reverse=True):
            if cache_file(args.smoke_cache, row["segment_id"]).is_file():
                smoke.append(row)
            if len(smoke) == 64:
                break
        if len(smoke) != 64:
            raise ValueError("need 64 cached original training examples for smoke")
        args.smoke_output.mkdir(parents=True, exist_ok=True)
        write_rows(args.smoke_output / "train.jsonl", smoke)
        write_rows(args.smoke_output / "dev.jsonl", splits["dev"])
        print(json.dumps({"smoke_rows": len(smoke), "smoke_output": str(args.smoke_output),
                          "selection": "longest question+answer characters among cached train rows"}))


if __name__ == "__main__":
    main()
