"""Train-only, answer-level distillation data; original references stay immutable."""
from collections import Counter
import random


def assert_disjoint(train, dev, evaluation):
    for field in ("question_id", "segment_id", "frames"):
        def values(rows):
            return {v for row in rows for v in (row[field] if field == "frames" else [row[field]])}
        groups = [values(rows) for rows in (train, dev, evaluation)]
        if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("train/dev/eval overlap in %s" % field)


def validate_predictions(originals, predictions):
    expected = {r["question_id"]: r for r in originals}
    actual = {r["question_id"]: r for r in predictions}
    if (not expected or len(expected) != len(originals) or len(actual) != len(predictions)
            or set(actual) != set(expected)):
        raise ValueError("prediction IDs are incomplete, duplicated or outside the split")
    for key, prediction in actual.items():
        row = expected[key]
        if (prediction["question"], prediction["segment_id"], prediction["references"]) != (
                row["question"], row["segment_id"], row["answers"]):
            raise ValueError("prediction does not match the original question/references")
    return actual


def select_training_rows(originals, student, teacher, replay_fraction=.3, seed=44):
    if not 0 < replay_fraction < 1:
        raise ValueError("replay_fraction must be between zero and one")
    s = validate_predictions(originals, student)
    t = validate_predictions(originals, teacher)
    wins, replay = [], []
    excluded = Counter()
    for row in originals:
        key = row["question_id"]
        if s[key]["lingo_judge_correct"]:
            replay.append(dict(row, distillation_source="train", target_source="reference_replay"))
        elif (t[key]["lingo_judge_correct"] and t[key].get("finished") is True
              and t[key]["prediction"].strip()):
            wins.append(dict(row, distillation_source="train", target_source="teacher",
                             teacher_answer=t[key]["prediction"].strip()))
        else:
            excluded["teacher_incorrect_or_unfinished"] += 1
    if not wins:
        raise ValueError("no complete teacher-correct/student-wrong training answers")
    rng = random.Random(seed)
    rng.shuffle(replay)
    # Every useful correction is retained. Replay is drawn from student-correct
    # cases, including those on which the teacher is wrong.
    replay_count = min(len(replay), round(len(wins) * replay_fraction / (1 - replay_fraction)))
    if replay_count == 0:
        raise ValueError("no student-correct examples available for replay")
    rows = wins + replay[:replay_count]
    rng.shuffle(rows)
    return rows, {"records": len(rows), "segments": len({r["segment_id"] for r in rows}),
                  "teacher_targets": len(wins), "reference_replay": replay_count,
                  "requested_replay_fraction": replay_fraction,
                  "actual_replay_fraction": replay_count / len(rows),
                  "source_counts": dict(Counter(r["source_subset"] for r in rows)),
                  "teacher_source_counts": dict(Counter(r["source_subset"] for r in wins)),
                  "excluded": dict(excluded), "seed": seed,
                  "supervision": "Judge-filtered offline teacher answers, not visual ground truth or OPD"}
