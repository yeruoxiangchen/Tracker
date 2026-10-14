#!/usr/bin/env python3
"""Filter teacher supervision using train-only paired predictions; preserve original QA."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl, dump_jsonl, file_sha256
from mpp_video_qa.observations import ANSWER_HINT_PROTOCOL


def select_reliable_rows(train, student, teacher, max_records, num_samples=4,
                         replay_fraction=0.3, seed=42):
    """RAIS-inspired train-only filtering; conditional teacher success is not visual accuracy."""
    if num_samples < 2 or max_records < 1 or not 0 <= replay_fraction < 1:
        raise ValueError("invalid reliability-selection configuration")
    originals = {r["question_id"]: r for r in train}
    if len(originals) != len(train):
        raise ValueError("duplicate training question IDs")

    def aggregate(predictions, context):
        groups = defaultdict(dict)
        for pred in predictions:
            qid, sample = pred["question_id"], pred["sample_id"]
            if qid not in originals:
                raise ValueError("prediction outside the training split")
            row = originals[qid]
            if (pred["segment_id"] != row["segment_id"] or pred["question"] != row["question"] or
                    pred["references"] != row["answers"] or pred.get("teacher_context") != context or
                    pred.get("view") != "full"):
                raise ValueError("probe question, reference or teacher context differs")
            if sample in groups[qid]:
                raise ValueError("duplicate (question_id, sample_id)")
            groups[qid][sample] = bool(pred["lingo_judge_correct"] and pred.get("finished", False))
        if any(set(draws) != set(range(num_samples)) for draws in groups.values()):
            raise ValueError("incomplete sampling group")
        return {qid: sum(draws.values()) / num_samples for qid, draws in groups.items()}

    s, t = aggregate(student, "view"), aggregate(teacher, "answer_hint")
    if set(s) != set(t) or set(s) != set(originals) or not s:
        raise ValueError("teacher and student must cover the complete nonempty training probe")
    wins, replay = [], []
    for qid in sorted(s):
        if t[qid] != 1.0:
            continue
        row = dict(originals[qid], opd_training_source="train",
                   opd_selection="both_correct" if s[qid] == 1.0 else "teacher_fixed",
                   teacher_success_rate=t[qid], student_success_rate=s[qid],
                   teacher_student_gap=t[qid] - s[qid], selection_num_samples=num_samples)
        (replay if s[qid] == 1.0 else wins).append(row)
    if not wins:
        raise ValueError("no reliable teacher/student gap; inspect probe results before training")
    rng = random.Random(seed)
    source_counts = Counter(originals[qid]["source_subset"] for qid in s)

    def balanced_pick(pool, size):
        pool = list(pool)
        rng.shuffle(pool)
        pool.sort(key=lambda r: r["teacher_student_gap"], reverse=True)
        counts, chosen, deferred = Counter(), [], []
        # Largest-remainder quotas preserve candidate source mix when eligible samples permit.
        target = {src: size * n / len(s) for src, n in source_counts.items()}
        quotas = {src: int(n) for src, n in target.items()}
        for src in sorted(target, key=lambda x: target[x] - quotas[x], reverse=True)[:size - sum(quotas.values())]:
            quotas[src] += 1
        for row in pool:
            src = row["source_subset"]
            if counts[src] < quotas[src]:
                chosen.append(row)
                counts[src] += 1
            else:
                deferred.append(row)
        return chosen + deferred[:max(0, size - len(chosen))]

    n = min(len(wins), max(1, round(max_records * (1 - replay_fraction))))
    selected = balanced_pick(wins, n)
    selected += balanced_pick(replay, min(len(replay), max_records - n,
                                          round(n * replay_fraction / (1 - replay_fraction))))
    rng.shuffle(selected)
    return selected, {
        "questions": len(s), "draws_per_model": len(student), "num_samples": num_samples,
        "student_success_rate_mean": sum(s.values()) / len(s),
        "conditional_teacher_success_rate_mean": sum(t.values()) / len(t),
        "reliable_teacher_questions": sum(v == 1.0 for v in t.values()),
        "eligible_teacher_fixed": len(wins), "eligible_both_correct": len(replay),
        "candidate_source_counts": dict(source_counts),
    }


def select_rows(train, student, teacher, max_records, replay_fraction=0.3, seed=42):
    if not 0 <= replay_fraction < 1 or max_records < 1:
        raise ValueError("invalid selection size/replay fraction")
    originals = {r["question_id"]: r for r in train}
    s, t = ({r["question_id"]: r for r in rows} for rows in (student, teacher))
    if len(s) != len(student) or len(t) != len(teacher) or set(s) != set(t):
        raise ValueError("student/teacher prediction IDs must be unique and identical")
    wins, replay = [], []
    for qid in sorted(t):
        if qid not in originals:
            raise ValueError("prediction is outside the training split: %s" % qid)
        row = originals[qid]
        for prediction in (s[qid], t[qid]):
            if (prediction["segment_id"] != row["segment_id"] or
                    prediction["question"] != row["question"] or prediction["references"] != row["answers"]):
                raise ValueError("training question/frames identity differs")
        if t[qid]["lingo_judge_correct"]:
            destination = replay if s[qid]["lingo_judge_correct"] else wins
            destination.append(dict(row, opd_training_source="train",
                                    opd_selection="both_correct" if destination is replay else "teacher_fixed"))
    if not wins:
        raise ValueError("no teacher-only correct training examples; no evidence for this OPD selection")
    rng = random.Random(seed)
    rng.shuffle(wins)
    rng.shuffle(replay)
    count = min(len(wins), max(1, round(max_records * (1 - replay_fraction))))
    selected = wins[:count] + replay[:min(max_records - count, round(count * replay_fraction / (1 - replay_fraction)))]
    rng.shuffle(selected)
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--student-scored", required=True)
    p.add_argument("--teacher-scored", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-records", type=int, default=1000)
    p.add_argument("--replay-fraction", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-samples", type=int, default=1)
    args = p.parse_args()
    train = load_jsonl(args.train)
    student, teacher = load_jsonl(args.student_scored), load_jsonl(args.teacher_scored)
    if args.num_samples > 1:
        selected, probe = select_reliable_rows(train, student, teacher, args.max_records,
                                               args.num_samples, args.replay_fraction, args.seed)
    else:
        selected = select_rows(train, student, teacher,
                               args.max_records, args.replay_fraction, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and load_jsonl(output) != selected:
        raise ValueError("existing OPD selection differs; use a new output file")
    identities = {}
    for name, scored in (("student", args.student_scored), ("teacher", args.teacher_scored)):
        manifest = Path(scored).parent.parent / "run_manifest.json"
        if not manifest.exists():
            raise ValueError("missing generation identity: %s" % manifest)
        summary = json.loads((Path(scored).parent / "summary.json").read_text())
        if summary.get("judge_protocol") != "official_lower_strip_maxref_logit_gt0_maxlength128_v1":
            raise ValueError("selection requires the corrected official Judge protocol")
        identities[name] = json.loads(manifest.read_text())
    if args.num_samples > 1:
        for name, context in (("student", "view"), ("teacher", "answer_hint")):
            ident = identities[name]
            if (ident.get("num_samples") != args.num_samples or not ident.get("sampling") or
                    ident.get("teacher_context") != context or ident.get("view") != "full" or
                    ident.get("data_sha256") != file_sha256(args.train)):
                raise ValueError("reliability selection requires complete stochastic full-frame train probes")
        if identities["teacher"].get("answer_hint_protocol") != ANSWER_HINT_PROTOCOL:
            raise ValueError("teacher answer-hint protocol mismatch")
    dump_jsonl(output, selected)
    manifest = {
        "records": len(selected), "segments": len({r["segment_id"] for r in selected}),
        "selection_counts": dict(Counter(r["opd_selection"] for r in selected)),
        "source_counts": dict(Counter(r["source_subset"] for r in selected)),
        "train_sha256": file_sha256(args.train), "selected_sha256": file_sha256(output),
        "student_scored_sha256": file_sha256(args.student_scored),
        "teacher_scored_sha256": file_sha256(args.teacher_scored),
        "generation_identities": identities,
        "filter": "teacher Judge correct; teacher-fixed preferred; both-correct replay",
        "limitation": "text Judge selection is not visual ground truth; inspect a sample"}
    if args.num_samples > 1:
        manifest.update(probe_sampling=probe, num_samples=args.num_samples,
                        filter="all K teacher draws correct and finished; rank student gap within source quotas",
                        limitation="Answer-conditioned teacher success measures label consistency, not independent visual ability. Judge can miss direction/action errors.")
    else:
        student_by_id = {row["question_id"]: row for row in student}
        fixed = sum(not student_by_id[row["question_id"]]["lingo_judge_correct"] and
                    row["lingo_judge_correct"] for row in teacher)
        regressed = sum(student_by_id[row["question_id"]]["lingo_judge_correct"] and
                        not row["lingo_judge_correct"] for row in teacher)
        manifest["probe_paired"] = {"records": len(teacher), "fixed": fixed,
                                    "regressed": regressed, "net": fixed - regressed}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
