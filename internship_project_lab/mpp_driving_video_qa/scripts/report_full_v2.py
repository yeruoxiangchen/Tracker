#!/usr/bin/env python3
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


METRICS = ("normalized_exact", "token_f1", "lingo_judge_correct")


def read(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return {row["question_id"]: row for row in rows}


def metric_mean(rows, name):
    values = [float(row[name]) for row in rows.values() if name in row]
    return statistics.mean(values) if values else None


def paired(base, candidate):
    common = sorted(set(base) & set(candidate))
    result = {"records": len(common)}
    for metric in METRICS:
        changes = [float(candidate[qid][metric]) - float(base[qid][metric]) for qid in common
                   if metric in base[qid] and metric in candidate[qid]]
        if changes:
            result[metric] = {"fixed": sum(value > 0 for value in changes),
                              "regressed": sum(value < 0 for value in changes),
                              "net_change_mean": statistics.mean(changes)}
    return result


def answer_prefixes(rows):
    prefixes = Counter(" ".join(row["prediction"].lower().split()[:3]) for row in rows.values())
    return [{"prefix": prefix, "count": count} for prefix, count in prefixes.most_common(15)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = {}
    for model in ("base", "old", "new"):
        for split in ("action", "scenery"):
            path = args.eval_root / ("%s_%s" % (model, split)) / "scored/scored_predictions.jsonl"
            if not path.is_file():
                raise FileNotFoundError("missing scored evaluation: %s" % path)
            loaded[(model, split)] = read(path)
    report = {"splits": {}, "paired_vs_base": {}, "action_prediction_prefixes": {}}
    for (model, split), rows in loaded.items():
        report["splits"].setdefault(split, {})[model] = {
            "records": len(rows), **{metric: metric_mean(rows, metric) for metric in METRICS}}
        if split == "action":
            report["action_prediction_prefixes"][model] = answer_prefixes(rows)
    for split in ("action", "scenery"):
        report["paired_vs_base"][split] = {
            model: paired(loaded[("base", split)], loaded[(model, split)])
            for model in ("old", "new")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
