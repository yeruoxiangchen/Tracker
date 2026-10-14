#!/usr/bin/env python3
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpp_video_qa.core import dump_jsonl, file_sha256, load_jsonl
from mpp_video_qa.schema import normalize_source_row


def read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else list(payload.values())


def parse_named_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return value.split("=", 1)


def parse_named_int(value):
    name, raw = parse_named_path(value)
    try:
        number = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected NAME=INTEGER") from error
    if number < 0:
        raise argparse.ArgumentTypeError("count must be non-negative")
    return name, number


def limited(rows, limit):
    return rows if limit == 0 else rows[:limit]


def source_counts(rows):
    questions = defaultdict(int)
    segments = defaultdict(set)
    for row in rows:
        source = row["source_subset"]
        questions[source] += 1
        segments[source].add(row["segment_id"])
    return {
        source: {"questions": questions[source], "segments": len(segments[source])}
        for source in sorted(questions)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", action="append", type=parse_named_path, required=True,
                        help="repeatable NAME=parquet_or_json")
    parser.add_argument("--eval-input", type=Path)
    parser.add_argument("--frames-root", type=Path,
                        help="default image root when a subset-specific root is not supplied")
    parser.add_argument("--frames-root-map", action="append", type=parse_named_path, default=[],
                        help="repeatable subset=/path/to/extracted/images")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--dev-segments", type=int, default=300)
    parser.add_argument("--max-qa-per-segment", type=int, default=12)
    parser.add_argument("--fixed-dev-data", action="append", type=parse_named_path, default=[],
                        help="repeatable SOURCE=JSONL; preserve these rows and exclude each "
                             "listed segment's complete raw group from training")
    parser.add_argument("--dev-segments-by-source", action="append", type=parse_named_int,
                        default=[], help="repeatable SOURCE=COUNT for additional held-out videos")
    parser.add_argument("--max-dev-qa-per-segment", type=int, default=12,
                        help="cap additional dev rows per video; 0 keeps every question")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()

    root_map = {name: Path(path) for name, path in args.frames_root_map}
    if args.frames_root is None and not root_map:
        parser.error("provide --frames-root or at least one --frames-root-map")
    rows = []
    rejected = defaultdict(int)
    for source, path in args.train_input:
        frames_root = root_map.get(source, args.frames_root)
        if frames_root is None:
            raise ValueError("no frame root configured for subset %s" % source)
        for raw in read_table(path):
            row = normalize_source_row(raw, source, frames_root)
            if row is None:
                rejected["missing_required_field"] += 1
                continue
            if args.check_files and not all(Path(p).is_file() for p in row["frames"]):
                rejected["missing_frame"] += 1
                continue
            rows.append(row)
    if args.train_size < 0 or args.max_qa_per_segment < 0 or args.max_dev_qa_per_segment < 0:
        parser.error("size and per-segment limits must be non-negative")

    grouped = defaultdict(list)
    segment_sources = defaultdict(set)
    for row in rows:
        key = (row["source_subset"], row["segment_id"])
        grouped[key].append(row)
        segment_sources[row["segment_id"]].add(row["source_subset"])
    collisions = {segment: sorted(sources) for segment, sources in segment_sources.items()
                  if len(sources) > 1}
    if collisions:
        example = next(iter(collisions.items()))
        raise RuntimeError("segment IDs collide across sources; cache identity would be unsafe: %s" %
                           (example,))

    rng = random.Random(args.seed)
    fixed_dev = []
    fixed_keys = set()
    for declared_source, path in args.fixed_dev_data:
        for row in load_jsonl(path):
            actual_source = row.get("source_subset", declared_source)
            if actual_source != declared_source:
                raise RuntimeError("fixed dev source mismatch in %s: %s != %s" %
                                   (path, actual_source, declared_source))
            key = (actual_source, row["segment_id"])
            if key not in grouped:
                raise RuntimeError("fixed dev segment is absent from raw input: %s" % (key,))
            if args.check_files and not all(Path(frame).is_file() for frame in row["frames"]):
                raise RuntimeError("fixed dev row has a missing frame: %s" % row["question_id"])
            fixed_keys.add(key)
            fixed_dev.append(row)

    extra_dev_keys = set()
    requested_by_source = dict(args.dev_segments_by_source)
    if requested_by_source:
        for source, count in sorted(requested_by_source.items()):
            candidates = sorted(key for key in grouped if key[0] == source and key not in fixed_keys)
            source_rng = random.Random("%d:%s" % (args.seed, source))
            source_rng.shuffle(candidates)
            if count > len(candidates):
                raise RuntimeError("requested %d dev segments for %s, only %d available" %
                                   (count, source, len(candidates)))
            extra_dev_keys.update(candidates[:count])
    elif not fixed_keys and args.dev_segments:
        candidates = sorted(grouped)
        rng.shuffle(candidates)
        extra_dev_keys.update(candidates[:args.dev_segments])

    train, extra_dev = [], []
    for key in sorted(grouped):
        samples = list(grouped[key])
        rng.shuffle(samples)
        if key in fixed_keys:
            continue
        if key in extra_dev_keys:
            extra_dev.extend(limited(samples, args.max_dev_qa_per_segment))
        else:
            train.extend(limited(samples, args.max_qa_per_segment))
    dev = fixed_dev + extra_dev
    rng.shuffle(train)
    rng.shuffle(dev)
    if args.train_size:
        train = train[:args.train_size]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.output_dir / "train.jsonl", train)
    dump_jsonl(args.output_dir / "dev.jsonl", dev)
    for source in sorted({row["source_subset"] for row in dev}):
        dump_jsonl(args.output_dir / ("dev_%s.jsonl" % source),
                   [row for row in dev if row["source_subset"] == source])

    eval_rows = []
    if args.eval_input:
        frames_root = root_map.get("evaluation", args.frames_root)
        if frames_root is None:
            raise ValueError("no frame root configured for evaluation subset")
        for raw in read_table(args.eval_input):
            row = normalize_source_row(raw, "evaluation", frames_root)
            if row is not None:
                eval_rows.append(row)
        # Official evaluation can contain one row per reference answer.
        merged = {}
        for row in eval_rows:
            key = (row["question_id"], row["segment_id"])
            if key not in merged:
                merged[key] = row
            else:
                merged[key]["answers"].extend(a for a in row["answers"]
                                               if a not in merged[key]["answers"])
        eval_rows = list(merged.values())
        dump_jsonl(args.output_dir / "eval.jsonl", eval_rows)

    train_keys = {(row["source_subset"], row["segment_id"]) for row in train}
    dev_keys = {(row["source_subset"], row["segment_id"]) for row in dev}
    eval_keys = {(row["source_subset"], row["segment_id"]) for row in eval_rows}
    train_paths = {path for row in train for path in row["frames"]}
    dev_paths = {path for row in dev for path in row["frames"]}
    eval_paths = {path for row in eval_rows for path in row["frames"]}
    overlaps = {
        "segment_train_dev": len(train_keys & dev_keys),
        "segment_train_eval": len(train_keys & eval_keys),
        "segment_dev_eval": len(dev_keys & eval_keys),
        "frame_train_dev": len(train_paths & dev_paths),
        "frame_train_eval": len(train_paths & eval_paths),
        "frame_dev_eval": len(dev_paths & eval_paths),
    }
    if any(overlaps.values()):
        raise RuntimeError("data split overlap detected: %s" % overlaps)
    all_rows = train + dev + eval_rows
    question_ids = [row["question_id"] for row in all_rows]
    if len(question_ids) != len(set(question_ids)):
        raise RuntimeError("question_id collision detected across output splits")
    segment_identity = defaultdict(set)
    for row in all_rows:
        segment_identity[row["segment_id"]].add(row["source_subset"])
    cross_source = {segment: sources for segment, sources in segment_identity.items()
                    if len(sources) > 1}
    if cross_source:
        raise RuntimeError("segment_id collision across sources would corrupt cache identity: %s" %
                           next(iter(cross_source.items())))
    report = {
        "train_questions": len(train), "train_segments": len({x["segment_id"] for x in train}),
        "dev_questions": len(dev), "dev_segments": len(dev_keys),
        "eval_questions": len(eval_rows), "rejected": dict(rejected),
        "source_counts": {"train": source_counts(train), "dev": source_counts(dev),
                          "eval": source_counts(eval_rows)},
        "fixed_dev_segments": len(fixed_keys),
        "additional_dev_segments": len(extra_dev_keys),
        "limits": {"train_size": args.train_size,
                   "max_qa_per_segment": args.max_qa_per_segment,
                   "max_dev_qa_per_segment": args.max_dev_qa_per_segment},
        "overlaps": overlaps,
        "inputs": {name: {"path": str(Path(path).resolve()),
                           "sha256": file_sha256(path)} for name, path in args.train_input},
        "outputs": {
            name: {"path": str((args.output_dir / (name + ".jsonl")).resolve()),
                   "sha256": file_sha256(args.output_dir / (name + ".jsonl"))}
            for name in ("train", "dev", "eval")
            if (args.output_dir / (name + ".jsonl")).is_file()
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
