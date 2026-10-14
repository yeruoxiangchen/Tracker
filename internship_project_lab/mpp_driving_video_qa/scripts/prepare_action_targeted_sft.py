#!/usr/bin/env python3
"""Build a train-only continuation set. Sampling tags are heuristics, not labels."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, dump_jsonl, file_sha256

# Only these reviewed question-level substitutions are used. Answers and frames
# are copied unchanged; polarity, time, and requested explanation are preserved.
PARAPHRASES = {
    "What are you paying attention to and why?": "What are you focusing on, and for what reason?",
    "Is it safe for you to accelerate in this situation?": "In this situation, is accelerating safe for you?",
    "Should you stop or brake in this situation?": "In this situation, should you brake or stop?",
    "What are you currently doing and why?": "What are you doing at present, and why?",
    "What is your current action and why?": "What action are you taking now, and for what reason?",
    "How should you adapt your speed in this situation?": "In this situation, how should you adjust your speed?",
    "Why is it unsafe for you to accelerate in this situation?": "What makes accelerating unsafe for you in this situation?",
    "Why are you decelerating in this situation?": "In this situation, why are you slowing down?",
    "What is the reason for decelerating?": "Why are you slowing down?",
    "Why are you positioned in this way?": "What is the reason for your current positioning?",
    "Why are you accelerating?": "What is the reason you are accelerating?",
    "Why are you decelerating?": "What is the reason you are slowing down?",
    "Why are you stopping?": "What is the reason you are stopping?",
    "What are you doing and why?": "What action are you taking, and why?",
    "Are there any hazards on the road ahead?": "Are any hazards present on the road ahead?",
    "What should you do in this situation?": "In this situation, what should you do?",
}


def tags_for(row):
    q = row["question"].lower()
    tags = []
    for tag, pattern in (
        ("speed", r"\b(accelerat\w*|decelerat\w*|speed|brak\w*|stop\w*|slow\w*)\b"),
        ("direction", r"\b(turn\w*|steer\w*|overtak\w*)\b"),
        ("position", r"\b(lane\w*|position\w*)\b"),
        ("current_action", r"\b(action|doing)\b"),
        ("reason_attention", r"\b(why|reason\w*|attention|hazard\w*)\b"),
    ):
        if re.search(pattern, q):
            tags.append(tag)
    return tags


def balanced_take(rows, count, rng, video_counts, question_counts, video_cap,
                  question_cap, group_fn):
    """Round-robin categories and question texts, without duplicating originals."""
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        buckets[group_fn(row)][row["question"]].append(row)
    queues = {}
    for group, questions in buckets.items():
        names = sorted(questions)
        rng.shuffle(names)
        qs = deque()
        for name in names:
            candidates = sorted(questions[name], key=lambda x: x["question_id"])
            rng.shuffle(candidates)
            qs.append(deque(candidates))
        queues[group] = qs
    groups = deque(sorted(queues))
    chosen = []
    while groups and len(chosen) < count:
        group = groups.popleft()
        questions = queues[group]
        found = None
        while questions and found is None:
            candidates = questions.popleft()
            while candidates:
                row = candidates.popleft()
                if (video_counts[row["segment_id"]] < video_cap and
                        question_counts[row["question"]] < question_cap):
                    found = row
                    break
            if candidates:
                questions.append(candidates)
        if found is not None:
            chosen.append(found)
            video_counts[found["segment_id"]] += 1
            question_counts[found["question"]] += 1
        if questions:
            groups.append(group)
    if len(chosen) != count:
        raise ValueError("requested %d rows but only %d fit the sampling limits" % (count, len(chosen)))
    return chosen


def build(args):
    source = args.source_dir
    train = load_jsonl(source / "train.jsonl")
    dev = load_jsonl(source / "dev.jsonl")
    evaluation = load_jsonl(source / "eval.jsonl")
    heldout = {r["segment_id"] for r in dev + evaluation}
    heldout_frames = {p for r in dev + evaluation for p in r["frames"]}
    if any(r["segment_id"] in heldout or heldout_frames.intersection(r["frames"]) for r in train):
        raise ValueError("train and held-out videos/frames overlap")
    if len({r["question_id"] for r in train}) != len(train):
        raise ValueError("duplicate source question IDs")
    if not 0 <= args.paraphrases <= args.focus:
        raise ValueError("paraphrases must be between zero and the focus count")
    rng = random.Random(args.seed)
    vc, qc = Counter(), Counter()
    focus_pool = [r for r in train if tags_for(r)]
    eligible = [r for r in focus_pool if r["question"] in PARAPHRASES]
    take = lambda rows, n, fn: balanced_take(
        rows, n, rng, vc, qc, args.max_originals_per_video,
        args.max_question_count, fn)
    group = lambda r: tags_for(r)[0]
    parents = take(eligible, args.paraphrases, group)
    used = {r["question_id"] for r in parents}
    focus = parents + take([r for r in focus_pool if r["question_id"] not in used],
                           args.focus - len(parents), group)
    used = {r["question_id"] for r in focus}
    replay = take([r for r in train if r["question_id"] not in used and not tags_for(r)],
                  args.replay, lambda r: r.get("source_subset", "unknown"))
    rows = []
    for kind, originals in (("focus_original", focus), ("replay", replay)):
        for row in originals:
            rows.append(dict(row, provenance={"kind": kind,
                "original_question_id": row["question_id"], "sampling_tags": tags_for(row)}))
    for row in parents:
        rows.append(dict(row, question_id=row["question_id"] + "::targeted_paraphrase_v1",
                         question=PARAPHRASES[row["question"]], provenance={
                             "kind": "paraphrase", "original_question_id": row["question_id"],
                             "original_question": row["question"], "sampling_tags": tags_for(row)}))
    rng.shuffle(rows)
    output = args.output_dir
    if (output / "manifest.json").exists() or (output / "train.jsonl").exists():
        raise ValueError("output already exists; use a new version directory")
    output.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output / "train.jsonl", rows)
    for name in ("dev", "dev_action", "dev_scenery", "eval"):
        # Copy complete existing split content, not a new holdout or selection.
        dump_jsonl(output / (name + ".jsonl"), load_jsonl(source / (name + ".jsonl")))
    audit_candidates = sorted(focus, key=lambda r: r["question_id"])
    rng.shuffle(audit_candidates)
    dump_jsonl(output / "review_sample.jsonl", audit_candidates[:args.review_size])
    manifest = {
        "protocol": "action_targeted_sft_v1", "seed": args.seed,
        "source_train_sha256": file_sha256(source / "train.jsonl"),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "counts": dict(Counter(r["provenance"]["kind"] for r in rows)),
        "source_counts": dict(Counter(r["source_subset"] for r in rows)),
        "tag_counts": dict(Counter(t for r in rows for t in r["provenance"]["sampling_tags"])),
        "videos": len({r["segment_id"] for r in rows}),
        "max_originals_per_video_observed": max(vc.values(), default=0),
        "max_rows_per_video_observed": max(Counter(r["segment_id"] for r in rows).values(), default=0),
        "max_original_question_count_observed": max(qc.values(), default=0),
        "paraphrase_templates": PARAPHRASES,
        "review_status": "sampling tags are heuristic; visual review sample exported, not yet reviewed",
        "outputs": {name: {"sha256": file_sha256(output / (name + ".jsonl"))}
                    for name in ("train", "dev", "dev_action", "dev_scenery", "eval")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, default=ROOT / "datasets/processed/lingoqa_full_v2")
    p.add_argument("--output-dir", type=Path, default=ROOT / "datasets/processed/lingoqa_action_targeted_v1")
    p.add_argument("--focus", type=int, default=5000)
    p.add_argument("--paraphrases", type=int, default=2000)
    p.add_argument("--replay", type=int, default=3000)
    p.add_argument("--max-originals-per-video", type=int, default=4)
    p.add_argument("--max-question-count", type=int, default=300)
    p.add_argument("--review-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.focus, args.paraphrases, args.replay, args.review_size) < 0:
        p.error("counts must be non-negative")
    build(args)


if __name__ == "__main__":
    main()
