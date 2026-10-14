#!/usr/bin/env python3
"""Original-QA SFT subset and the fixed 80-case teacher diagnostic."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, dump_jsonl, file_sha256


def stratified_subset(rows, size, seed=42, per_segment=4):
    if not 0 < size <= len(rows):
        raise ValueError("train-size must be within the source dataset")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_subset"]].append(row)
    sources = sorted(groups)
    allocations = {source: int(size * len(groups[source]) / len(rows)) for source in sources}
    for source in sorted(sources, key=lambda s: size * len(groups[s]) / len(rows) - allocations[s],
                         reverse=True)[:size - sum(allocations.values())]:
        allocations[source] += 1
    chosen = []
    for source in sources:
        by_segment = defaultdict(list)
        for row in groups[source]:
            by_segment[row["segment_id"]].append(row)
        segments = sorted(by_segment)
        rng.shuffle(segments)
        for segment in segments:
            rng.shuffle(by_segment[segment])
        pool = [by_segment[segment][round_index]
                for round_index in range(per_segment) for segment in segments
                if len(by_segment[segment]) > round_index]
        if len(pool) < allocations[source]:
            raise ValueError("per-segment cap cannot supply %s quota" % source)
        chosen.extend(pool[:allocations[source]])
    rng.shuffle(chosen)
    return chosen


def write_rows(path, rows):
    if path.exists():
        if load_jsonl(path) != rows:
            raise ValueError("existing dataset differs; choose a new output directory: %s" % path)
        return
    dump_jsonl(path, rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", default=str(ROOT / "datasets/processed/lingoqa_full_v2"))
    p.add_argument("--output-dir", default=str(ROOT / "datasets/processed/lingoqa_visual20k_v1"))
    p.add_argument("--cases", default=str(ROOT / "outputs/action_targeted_diagnosis_v1/cases.jsonl"))
    p.add_argument("--baseline-dir", default=str(ROOT / "outputs/projector_sft_eval"))
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--teacher-train-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-qa-per-segment", type=int, default=4)
    args = p.parse_args()
    source, output = Path(args.source_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = load_jsonl(source / "train.jsonl")
    dev, evaluation = load_jsonl(source / "dev.jsonl"), load_jsonl(source / "eval.jsonl")
    held_out = {row["segment_id"] for row in dev + evaluation}
    if held_out & {row["segment_id"] for row in train}:
        raise ValueError("train overlaps dev/evaluation segments")
    selected = stratified_subset(train, args.train_size, args.seed, args.max_qa_per_segment)
    cases = load_jsonl(args.cases)
    dev_by_id = {row["question_id"]: row for row in dev}
    if len(cases) != 48 or len({r["segment_id"] for r in cases}) != 48:
        raise ValueError("expected the original 48 diagnostic videos")
    diagnostic = []
    for row in cases:
        original = dev_by_id[row["question_id"]]
        if any(row[key] != original[key] for key in ("frames", "question", "answers", "segment_id")):
            raise ValueError("diagnostic metadata differs from dev")
        diagnostic.append(dict(original, diagnostic_group="diagnostic48"))
    existing_segments = {row["segment_id"] for row in diagnostic}
    baseline = []
    for split in ("action", "scenery"):
        baseline.extend(load_jsonl(Path(args.baseline_dir) / ("joint_" + split) /
                                   "scored/scored_predictions.jsonl"))
    rng = random.Random(args.seed)
    rng.shuffle(baseline)
    controls = []
    for scored in baseline:
        row = dev_by_id[scored["question_id"]]
        if scored.get("lingo_judge_correct") and row["segment_id"] not in existing_segments:
            controls.append(dict(row, diagnostic_group="correct_control32"))
            existing_segments.add(row["segment_id"])
            if len(controls) == 32:
                break
    if len(controls) != 32:
        raise ValueError("not enough distinct correct control videos")
    for name, rows in (("train", selected), ("dev", dev), ("eval", evaluation),
                       ("teacher_probe", diagnostic + controls),
                       ("teacher_train_probe", selected[:args.teacher_train_size])):
        write_rows(output / (name + ".jsonl"), rows)
    for name in ("dev_action", "dev_scenery"):
        write_rows(output / (name + ".jsonl"), load_jsonl(source / (name + ".jsonl")))
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
                "records": len(selected), "segments": len({r["segment_id"] for r in selected}),
                "source_counts": dict(Counter(r["source_subset"] for r in selected)),
                "max_qa_per_segment": args.max_qa_per_segment,
                "source_train_sha256": file_sha256(source / "train.jsonl"),
                "train_sha256": file_sha256(output / "train.jsonl"),
                "teacher_probe_records": 80, "teacher_probe_scope": "dev only; not training",
                "question_and_answers": "unchanged originals"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
